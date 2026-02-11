import io
import math
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional


import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageDraw

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector

try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    _CAPTION_OK = True
except Exception:
    _CAPTION_OK = False


st.set_page_config(page_title="Sprite Animator (Pose-ControlNet)", layout="wide")


@dataclass
class SizePlan:
    base_w: int
    base_h: int
    gen_w: int
    gen_h: int
    mode: str


def calc_gen_size(w: int, h: int, mode: str) -> SizePlan:
    def floor64(x: int) -> int:
        return max(64, (x // 64) * 64)

    def ceil64(x: int) -> int:
        return max(64, ((x + 63) // 64) * 64)

    if w % 64 == 0 and h % 64 == 0:
        return SizePlan(w, h, w, h, mode)

    if mode == "resize_down":
        return SizePlan(w, h, floor64(w), floor64(h), mode)
    if mode == "resize_up":
        return SizePlan(w, h, ceil64(w), ceil64(h), mode)
    return SizePlan(w, h, ceil64(w), ceil64(h), mode)


def resize_or_pad(img: Image.Image, plan: SizePlan) -> Image.Image:
    if plan.gen_w == plan.base_w and plan.gen_h == plan.base_h:
        return img

    if plan.mode == "pad":
        pad_w = plan.gen_w - plan.base_w
        pad_h = plan.gen_h - plan.base_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=0)

    return img.resize((plan.gen_w, plan.gen_h), Image.LANCZOS)


def crop_or_resize_back(img: Image.Image, plan: SizePlan) -> Image.Image:
    if plan.gen_w == plan.base_w and plan.gen_h == plan.base_h:
        return img

    if plan.mode == "pad":
        left = (plan.gen_w - plan.base_w) // 2
        top = (plan.gen_h - plan.base_h) // 2
        return img.crop((left, top, left + plan.base_w, top + plan.base_h))

    return img.resize((plan.base_w, plan.base_h), Image.LANCZOS)


@st.cache_resource
def load_openpose() -> OpenposeDetector:
    return OpenposeDetector.from_pretrained("lllyasviel/Annotators")


@st.cache_resource
def load_pipe(base_model_id: str, controlnet_id: str, use_cuda: bool):
    dtype = torch.float16 if use_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if use_cuda:
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to("cuda")
    else:
        pipe.enable_attention_slicing()
        pipe.to("cpu")

    return pipe


@st.cache_resource
def load_captioner():
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    processor = ViTImageProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer


def caption_image(img: Image.Image) -> str:
    model, processor, tokenizer = load_captioner()
    pixel_values = processor(images=[img], return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=24, num_beams=4)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def generate_pose_sequence(pose_img: Image.Image, n_frames: int, jitter: int, rotate_deg: float, bob_px: int) -> List[Image.Image]:
    poses = []
    w, h = pose_img.size

    for i in range(n_frames):
        t = (i / max(1, n_frames - 1)) * 2 * math.pi
        dx = int(math.sin(t) * jitter)
        dy = int(math.sin(t + math.pi / 2) * bob_px)
        angle = math.sin(t) * rotate_deg

        img = pose_img.rotate(angle, resample=Image.BILINEAR, expand=False)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(img, (dx, dy))
        poses.append(canvas)

    return poses


def blend_pose(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return Image.blend(a, b, max(0.0, min(1.0, t)))


def pil_from_upload(uploaded) -> Image.Image:
    return Image.open(uploaded).convert("RGB")


def pose_has_signal(pose_img: Image.Image) -> bool:
    # OpenPose output is mostly black when no body is detected.
    arr = np.array(pose_img.convert("RGB"))
    non_black = np.any(arr > 12, axis=2)
    return float(non_black.mean()) > 0.005


def preprocess_for_pose(img: Image.Image, target_min_side: int, contrast_boost: float) -> Image.Image:
    w, h = img.size
    min_side = max(1, min(w, h))
    scale = max(1.0, float(target_min_side) / float(min_side))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    up = img.resize((new_w, new_h), Image.NEAREST)
    if contrast_boost > 1.0:
        up = ImageEnhance.Contrast(up).enhance(contrast_boost)
    return up


def extract_pose_with_fallback(
    img: Image.Image,
    openpose: OpenposeDetector,
    target_min_side: int,
    contrast_boost: float,
    use_hands_face: bool,
) -> Optional[Image.Image]:
    candidates = [
        preprocess_for_pose(img, target_min_side, contrast_boost),
        preprocess_for_pose(img, target_min_side * 2, contrast_boost),
        preprocess_for_pose(img, target_min_side * 2, max(1.0, contrast_boost + 0.4)),
    ]
    for cand in candidates:
        pose = openpose(cand, hand_and_face=use_hands_face)
        if pose_has_signal(pose):
            return pose.resize(img.size, Image.NEAREST)
    return None


def generate_bbox_pose(img: Image.Image) -> Optional[Image.Image]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        # fallback for non-alpha sprites: use non-black pixels
        rgb = np.array(img.convert("RGB"))
        mask = np.any(rgb > 15, axis=2)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = img.size
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)

    cx = x0 + bw // 2
    head_r = max(2, int(bh * 0.10))
    head_y = y0 + max(head_r + 1, int(bh * 0.12))
    neck_y = head_y + head_r + max(1, int(bh * 0.04))
    hip_y = y0 + int(bh * 0.58)
    shoulder_span = max(4, int(bw * 0.45))
    hip_span = max(4, int(bw * 0.30))
    leg_len = max(6, int(bh * 0.30))
    arm_len = max(5, int(bh * 0.20))

    ls = (cx - shoulder_span // 2, neck_y)
    rs = (cx + shoulder_span // 2, neck_y)
    lh = (cx - hip_span // 2, hip_y)
    rh = (cx + hip_span // 2, hip_y)
    l_elbow = (ls[0] - int(arm_len * 0.5), neck_y + int(arm_len * 0.5))
    r_elbow = (rs[0] + int(arm_len * 0.5), neck_y + int(arm_len * 0.5))
    l_hand = (ls[0] - arm_len, neck_y + arm_len)
    r_hand = (rs[0] + arm_len, neck_y + arm_len)
    l_foot = (lh[0] - max(1, int(bw * 0.08)), hip_y + leg_len)
    r_foot = (rh[0] + max(1, int(bw * 0.08)), hip_y + leg_len)

    pose = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(pose)
    lw = max(2, int(min(w, h) * 0.01))

    # head
    d.ellipse((cx - head_r, head_y - head_r, cx + head_r, head_y + head_r), outline=(255, 255, 255), width=lw)
    # torso
    d.line([(cx, neck_y), (cx, hip_y)], fill=(255, 255, 255), width=lw)
    # shoulders/hips
    d.line([ls, rs], fill=(255, 255, 255), width=lw)
    d.line([lh, rh], fill=(255, 255, 255), width=lw)
    # arms
    d.line([ls, l_elbow, l_hand], fill=(255, 255, 255), width=lw)
    d.line([rs, r_elbow, r_hand], fill=(255, 255, 255), width=lw)
    # legs
    d.line([lh, l_foot], fill=(255, 255, 255), width=lw)
    d.line([rh, r_foot], fill=(255, 255, 255), width=lw)

    return pose


def apply_chroma_key(img: Image.Image, key_rgb: tuple, tol: int) -> Image.Image:
    rgba = img.convert("RGBA")
    data = np.array(rgba)
    r, g, b, a = data.T
    kr, kg, kb = key_rgb
    mask = (
        (np.abs(r - kr) <= tol) &
        (np.abs(g - kg) <= tol) &
        (np.abs(b - kb) <= tol)
    )
    data[..., 3] = np.where(mask, 0, data[..., 3])
    return Image.fromarray(data)


def recenter_on_canvas(img: Image.Image, target_w: int, target_h: int, dx: int, dy: int) -> Image.Image:
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    x = (target_w - img.width) // 2 + dx
    y = (target_h - img.height) // 2 + dy
    canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
    return canvas


def crop_image(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    return img.crop((x, y, x + w, y + h))


def zip_frames(frames: List[Image.Image]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(frames):
            b = io.BytesIO()
            img.save(b, format="PNG")
            zf.writestr(f"frame_{i+1:03d}.png", b.getvalue())
    return buf.getvalue()


def ensure_frames(n: int):
    if "frames" not in st.session_state:
        st.session_state.frames = []
    frames = st.session_state.frames
    while len(frames) < n:
        frames.append({"image": None, "pose": None, "caption": ""})
    while len(frames) > n:
        frames.pop()


def first_keyframe_index() -> Optional[int]:
    for i, fr in enumerate(st.session_state.frames):
        if fr["image"] is not None:
            return i
    return None


st.title("Sprite Animator (pose-guided)")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1) Configurazione")
    n_frames = st.slider("Numero fotogrammi", min_value=2, max_value=24, value=8)
    ensure_frames(n_frames)

    base_model_id = st.text_input("Base model (SD 1.5)", value="runwayml/stable-diffusion-v1-5")
    controlnet_id = st.text_input("ControlNet OpenPose", value="lllyasviel/control_v11p_sd15_openpose")

    use_gpu = st.checkbox("Usa GPU se disponibile", value=True)
    use_cuda = torch.cuda.is_available() and use_gpu
    st.caption(f"CUDA disponibile: {torch.cuda.is_available()} | Device attivo: {'cuda' if use_cuda else 'cpu'}")

    st.subheader("2) Prompt")
    prompt = st.text_area("Prompt animazione", value="personaggio che cammina", height=80)
    negative = st.text_input("Negative prompt", value="low quality, bad anatomy, blurry")

    steps = st.slider("Steps", min_value=10, max_value=40, value=20)
    guidance = st.slider("CFG", min_value=1.0, max_value=12.0, value=7.5)
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42)

    st.subheader("3) Pose")
    pose_strength = st.slider("Jitter laterale (px)", 0, 30, 6)
    pose_bob = st.slider("Bobbing verticale (px)", 0, 30, 4)
    pose_rotate = st.slider("Rotazione (gradi)", 0.0, 15.0, 3.0)
    pose_min_side = st.slider("Risoluzione minima per estrazione pose", 256, 1536, 768, step=64)
    pose_contrast = st.slider("Boost contrasto (estrazione pose)", 1.0, 2.5, 1.4, step=0.1)
    pose_hands_face = st.checkbox("Rileva mani e viso", value=False)
    pose_bbox_fallback = st.checkbox("Fallback: crea scheletro base se OpenPose fallisce", value=True)

    size_mode = st.selectbox(
        "Gestione dimensioni (multipli di 64)",
        ["resize_down", "resize_up", "pad"],
        index=0,
        help="resize_down = meno token ma meno dettaglio; pad = mantiene dimensione ma aumenta token"
    )

    st.subheader("4) Descrizione frame")
    enable_caption = st.checkbox("Genera descrizione automatica (scarica modello)", value=False, disabled=not _CAPTION_OK)
    if not _CAPTION_OK:
        st.caption("Per le descrizioni automatiche serve transformers; disattivato.")

with col_right:
    st.subheader("Anteprima livello")
    show_sprites = st.checkbox("Mostra sprite", value=True)
    show_poses = st.checkbox("Mostra scheletro", value=True)
    frame_index_preview = st.slider("Frame da visualizzare", min_value=1, max_value=n_frames, value=1)

    fr = st.session_state.frames[frame_index_preview - 1]
    if show_sprites and fr["image"] is not None:
        st.image(fr["image"], caption=f"Sprite frame {frame_index_preview}")
    if show_poses and fr["pose"] is not None:
        st.image(fr["pose"], caption=f"Pose frame {frame_index_preview}")


st.subheader("5) Frame (carica le immagini chiave)")

cols = st.columns(4)
for i in range(n_frames):
    with cols[i % 4]:
        st.markdown(f"**Frame {i+1}**")
        up = st.file_uploader(
            "Carica immagine frame",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"frame_upload_{i}",
            label_visibility="collapsed",
        )
        if up:
            img = pil_from_upload(up)
            openpose = load_openpose()
            pose = extract_pose_with_fallback(
                img,
                openpose,
                target_min_side=pose_min_side,
                contrast_boost=pose_contrast,
                use_hands_face=pose_hands_face,
            )
            cap = ""
            if enable_caption:
                cap = caption_image(img)
            st.session_state.frames[i]["image"] = img
            if pose is None and pose_bbox_fallback:
                pose = generate_bbox_pose(img)
            st.session_state.frames[i]["pose"] = pose
            st.session_state.frames[i]["caption"] = cap
            if pose is None:
                st.warning(
                    f"Frame {i+1}: pose non rilevata. Prova a aumentare 'Risoluzione minima', "
                    "alzare contrasto, usare sprite piu grande o attivare fallback scheletro base."
                )

        if st.session_state.frames[i]["image"] is not None:
            st.image(st.session_state.frames[i]["image"], caption="Sprite", width=200)
            if st.session_state.frames[i]["pose"] is not None:
                st.image(st.session_state.frames[i]["pose"], caption="Pose", width=200)
            if st.session_state.frames[i]["caption"]:
                st.caption(st.session_state.frames[i]["caption"])
            if st.button("Rimuovi", key=f"remove_{i}"):
                st.session_state.frames[i] = {"image": None, "pose": None, "caption": ""}


st.subheader("6) Strumenti frame (ritaglio, ricentratura, trasparenza)")
edit_img_idx = st.selectbox("Scegli frame immagine", options=list(range(1, n_frames + 1)), key="edit_img_idx")
fr_img = st.session_state.frames[edit_img_idx - 1]

if fr_img["image"] is None:
    st.info("Il frame selezionato non ha un'immagine. Carica uno sprite nel frame.")
else:
    img_base = fr_img["image"]
    st.image(img_base, caption=f"Sprite originale {img_base.width}x{img_base.height}", width=240)

    st.markdown("**Risoluzione target**")
    target_w = st.number_input("Target W", min_value=16, max_value=4096, value=img_base.width, step=1, key="target_w")
    target_h = st.number_input("Target H", min_value=16, max_value=4096, value=img_base.height, step=1, key="target_h")

    st.markdown("**Ritaglio da immagine più grande**")
    max_x = max(0, img_base.width - 1)
    max_y = max(0, img_base.height - 1)
    crop_x = st.slider("Crop X", 0, max_x, 0, key="crop_x")
    crop_y = st.slider("Crop Y", 0, max_y, 0, key="crop_y")
    crop_w = st.slider("Crop W", 16, img_base.width - crop_x, img_base.width - crop_x, key="crop_w")
    crop_h = st.slider("Crop H", 16, img_base.height - crop_y, img_base.height - crop_y, key="crop_h")

    st.markdown("**Ricentratura**")
    move_x = st.slider("Sposta X (px)", -200, 200, 0, key="move_x")
    move_y = st.slider("Sposta Y (px)", -200, 200, 0, key="move_y")

    st.markdown("**Colore da rendere trasparente (chroma key)**")
    key_color = st.color_picker("Scegli colore", value="#00ff00", key="key_color")
    key_tol = st.slider("Tolleranza", 0, 60, 10, key="key_tol")

    if st.button("Applica modifiche immagine"):
        img = img_base
        img = crop_image(img, crop_x, crop_y, crop_w, crop_h)
        img = img.resize((target_w, target_h), Image.LANCZOS)

        # chroma key
        kr = int(key_color[1:3], 16)
        kg = int(key_color[3:5], 16)
        kb = int(key_color[5:7], 16)
        img = apply_chroma_key(img, (kr, kg, kb), key_tol)

        # ricentratura su canvas target
        img = recenter_on_canvas(img, target_w, target_h, move_x, move_y)

        fr_img["image"] = img.convert("RGBA")
        openpose = load_openpose()
        fr_img["pose"] = extract_pose_with_fallback(
            img.convert("RGB"),
            openpose,
            target_min_side=pose_min_side,
            contrast_boost=pose_contrast,
            use_hands_face=pose_hands_face,
        )
        if fr_img["pose"] is None and pose_bbox_fallback:
            fr_img["pose"] = generate_bbox_pose(img)
        st.session_state.frames[edit_img_idx - 1] = fr_img
        if fr_img["pose"] is None:
            st.warning("Pose non rilevata dopo le modifiche. Prova con sprite piu grande o contrasto piu alto.")

    st.caption("Nota: la pose viene ricalcolata dopo ogni modifica.")


st.subheader("7) Editor pose (manuale)")
edit_idx = st.selectbox("Scegli frame da modificare", options=list(range(1, n_frames + 1)))
fr_edit = st.session_state.frames[edit_idx - 1]

if fr_edit["pose"] is None:
    st.info("Il frame selezionato non ha una pose. Carica un'immagine o genera le pose.")
else:
    st.image(fr_edit["pose"], caption=f"Pose frame {edit_idx}", width=240)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        dx = st.slider("Sposta X (px)", -50, 50, 0)
        dy = st.slider("Sposta Y (px)", -50, 50, 0)
    with col_b:
        rot = st.slider("Ruota (gradi)", -20.0, 20.0, 0.0)
    with col_c:
        scale = st.slider("Scala (%)", 50, 150, 100)

    if st.button("Applica trasformazione"):
        img = fr_edit["pose"]
        w, h = img.size
        img2 = img.resize((int(w * scale / 100), int(h * scale / 100)), Image.BILINEAR)
        img2 = img2.rotate(rot, resample=Image.BILINEAR, expand=False)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        x = (w - img2.size[0]) // 2 + dx
        y = (h - img2.size[1]) // 2 + dy
        canvas.paste(img2, (x, y))
        fr_edit["pose"] = canvas
        st.session_state.frames[edit_idx - 1] = fr_edit

    st.caption("Suggerimento: puoi sostituire la pose con un file modificato esternamente.")
    pose_upload = st.file_uploader("Carica pose modificata (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"], key=f"pose_upload_{edit_idx}")
    if pose_upload:
        fr_edit["pose"] = pil_from_upload(pose_upload)
        st.session_state.frames[edit_idx - 1] = fr_edit

    if st.button("Svuota pose frame"):
        fr_edit["pose"] = None
        st.session_state.frames[edit_idx - 1] = fr_edit


st.subheader("8) Generazione pose per frame vuoti")

a = st.button("Genera/riempi pose mancanti")
if a:
    idx0 = first_keyframe_index()
    if idx0 is None:
        st.error("Carica almeno un frame chiave.")
    else:
        base_pose = st.session_state.frames[idx0]["pose"]
        if base_pose is None:
            st.error("Il frame chiave non ha pose. Ricarica l'immagine.")
        else:
            auto_seq = generate_pose_sequence(base_pose, n_frames, pose_strength, pose_rotate, pose_bob)

            # usa keyframe come vincoli e interpola tra loro
            key_idxs = [i for i, fr in enumerate(st.session_state.frames) if fr["pose"] is not None]
            for i in range(n_frames):
                if st.session_state.frames[i]["pose"] is None:
                    # trova keyframe precedente e successivo
                    prev = max([k for k in key_idxs if k < i], default=None)
                    nxt = min([k for k in key_idxs if k > i], default=None)
                    if prev is not None and nxt is not None:
                        t = (i - prev) / (nxt - prev)
                        pose = blend_pose(st.session_state.frames[prev]["pose"], st.session_state.frames[nxt]["pose"], t)
                    else:
                        pose = auto_seq[i]
                    st.session_state.frames[i]["pose"] = pose


st.subheader("9) Generazione immagini finali")

run_btn = st.button("Genera fotogrammi", type="primary")

if run_btn:
    # base sprite = primo frame con immagine
    idx0 = first_keyframe_index()
    if idx0 is None:
        st.error("Carica almeno un frame con immagine.")
        st.stop()

    base_img = st.session_state.frames[idx0]["image"]
    plan = calc_gen_size(base_img.size[0], base_img.size[1], size_mode)

    # verifica pose per tutti i frame
    for i, fr in enumerate(st.session_state.frames):
        if fr["pose"] is None:
            st.error(f"Manca la pose nel frame {i+1}. Usa 'Genera/riempi pose mancanti'.")
            st.stop()

    with st.spinner("Caricamento modelli..."):
        pipe = load_pipe(base_model_id, controlnet_id, use_cuda)

    results: List[Image.Image] = []
    prog = st.progress(0)
    status = st.empty()

    generator = torch.Generator(device="cuda" if use_cuda else "cpu").manual_seed(int(seed))

    for i, fr in enumerate(st.session_state.frames):
        status.text(f"Generazione frame {i+1}/{n_frames}")
        prog.progress(int((i / n_frames) * 100))

        pose_in = resize_or_pad(fr["pose"], plan)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=pose_in,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            width=plan.gen_w,
            height=plan.gen_h,
        ).images[0]

        image = crop_or_resize_back(image, plan)
        results.append(image)

    prog.progress(100)
    status.text("Completato")

    st.subheader("Risultati")
    st.image(results, caption=[f"Frame {i+1}" for i in range(len(results))], width=200)

    zip_bytes = zip_frames(results)
    st.download_button("Scarica frames (ZIP)", data=zip_bytes, file_name="frames.zip")
