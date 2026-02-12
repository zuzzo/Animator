import io
import copy
import math
import sys
import zipfile
import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter
from PySide6.QtCore import QMimeData, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QDrag, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

    CAPTION_OK = True
except Exception:
    CAPTION_OK = False


@dataclass
class SizePlan:
    base_w: int
    base_h: int
    gen_w: int
    gen_h: int
    mode: str


@dataclass
class FrameData:
    image: Optional[Image.Image] = None
    image_offset: Tuple[int, int] = (0, 0)
    rig_nodes: Optional[Dict[str, Tuple[float, float]]] = None
    bone_lengths: Optional[Dict[Tuple[str, str], float]] = None
    is_keyframe: bool = False
    key_description: str = ""
    caption: str = ""
    rig_profile: str = "auto"  # auto | human | animal | object
    detected_profile: str = "human"
    generated: Optional[Image.Image] = None


HUMAN_BONES = [
    ("neck", "head"),
    ("neck", "l_shoulder"),
    ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_hand"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_hand"),
    ("neck", "hip"),
    ("hip", "l_knee"),
    ("l_knee", "l_foot"),
    ("hip", "r_knee"),
    ("r_knee", "r_foot"),
]

ANIMAL_BONES = [
    ("head", "neck"),
    ("neck", "spine"),
    ("spine", "hip"),
    ("neck", "l_front_knee"),
    ("l_front_knee", "l_front_foot"),
    ("neck", "r_front_knee"),
    ("r_front_knee", "r_front_foot"),
    ("hip", "l_back_knee"),
    ("l_back_knee", "l_back_foot"),
    ("hip", "r_back_knee"),
    ("r_back_knee", "r_back_foot"),
    ("hip", "tail"),
]

OBJECT_BONES = [
    ("center", "top"),
    ("center", "bottom"),
    ("center", "left"),
    ("center", "right"),
    ("center", "pivot"),
]

RIG_BONES_BY_PROFILE = {
    "human": HUMAN_BONES,
    "animal": ANIMAL_BONES,
    "object": OBJECT_BONES,
}

RIG_COLORS = {
    "left": (34, 197, 94),
    "right": (59, 130, 246),
    "center": (249, 115, 22),
}


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qimg = QImage.fromData(buf.getvalue(), "PNG")
    return QPixmap.fromImage(qimg)


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


def pose_has_signal(pose_img: Image.Image) -> bool:
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


def generate_bbox_rig(img: Image.Image) -> Optional[Dict[str, Tuple[float, float]]]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
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

    ls = (float(cx - shoulder_span // 2), float(neck_y))
    rs = (float(cx + shoulder_span // 2), float(neck_y))
    lh = (float(cx - hip_span // 2), float(hip_y))
    rh = (float(cx + hip_span // 2), float(hip_y))
    l_elbow = (float(ls[0] - int(arm_len * 0.5)), float(neck_y + int(arm_len * 0.5)))
    r_elbow = (float(rs[0] + int(arm_len * 0.5)), float(neck_y + int(arm_len * 0.5)))
    l_hand = (float(ls[0] - arm_len), float(neck_y + arm_len))
    r_hand = (float(rs[0] + arm_len), float(neck_y + arm_len))
    l_knee = (float(lh[0]), float(hip_y + leg_len * 0.55))
    r_knee = (float(rh[0]), float(hip_y + leg_len * 0.55))
    l_foot = (float(lh[0] - max(1, int(bw * 0.08))), float(hip_y + leg_len))
    r_foot = (float(rh[0] + max(1, int(bw * 0.08))), float(hip_y + leg_len))
    return {
        "head": (float(cx), float(head_y)),
        "neck": (float(cx), float(neck_y)),
        "l_shoulder": ls,
        "r_shoulder": rs,
        "l_elbow": l_elbow,
        "r_elbow": r_elbow,
        "l_hand": l_hand,
        "r_hand": r_hand,
        "hip": (float(cx), float(hip_y)),
        "l_knee": l_knee,
        "r_knee": r_knee,
        "l_foot": l_foot,
        "r_foot": r_foot,
    }


def generate_animal_rig(img: Image.Image) -> Optional[Dict[str, Tuple[float, float]]]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        rgb = np.array(img.convert("RGB"))
        mask = np.any(rgb > 15, axis=2)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    # generic quadruped layout
    neck = (float(x0 + bw * 0.35), float(y0 + bh * 0.40))
    head = (float(x0 + bw * 0.20), float(y0 + bh * 0.30))
    spine = (float(x0 + bw * 0.55), float(y0 + bh * 0.45))
    hip = (float(x0 + bw * 0.72), float(y0 + bh * 0.50))
    return {
        "head": head,
        "neck": neck,
        "spine": spine,
        "hip": hip,
        "l_front_knee": (float(x0 + bw * 0.33), float(y0 + bh * 0.65)),
        "l_front_foot": (float(x0 + bw * 0.30), float(y1)),
        "r_front_knee": (float(x0 + bw * 0.42), float(y0 + bh * 0.66)),
        "r_front_foot": (float(x0 + bw * 0.45), float(y1)),
        "l_back_knee": (float(x0 + bw * 0.68), float(y0 + bh * 0.67)),
        "l_back_foot": (float(x0 + bw * 0.66), float(y1)),
        "r_back_knee": (float(x0 + bw * 0.78), float(y0 + bh * 0.68)),
        "r_back_foot": (float(x0 + bw * 0.80), float(y1)),
        "tail": (float(x0 + bw * 0.92), float(y0 + bh * 0.40)),
    }


def generate_object_rig(img: Image.Image) -> Optional[Dict[str, Tuple[float, float]]]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        rgb = np.array(img.convert("RGB"))
        mask = np.any(rgb > 15, axis=2)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return {
        "center": (cx, cy),
        "top": (cx, y0),
        "bottom": (cx, y1),
        "left": (x0, cy),
        "right": (x1, cy),
        "pivot": (cx, cy),
    }


def infer_subject_profile(caption: str) -> str:
    text = (caption or "").lower()
    animal_kw = ["dog", "cat", "horse", "bird", "wolf", "animal", "fox", "lion", "tiger", "deer", "bear"]
    human_kw = ["person", "man", "woman", "boy", "girl", "human", "character", "people"]
    object_kw = ["car", "ball", "box", "sword", "gun", "ship", "vehicle", "object", "robot", "item", "tool"]
    if any(k in text for k in animal_kw):
        return "animal"
    if any(k in text for k in human_kw):
        return "human"
    if any(k in text for k in object_kw):
        return "object"
    return "human"


def compute_bone_lengths(nodes: Dict[str, Tuple[float, float]], bones: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    out = {}
    for a, b in bones:
        if a not in nodes or b not in nodes:
            continue
        ax, ay = nodes[a]
        bx, by = nodes[b]
        out[(a, b)] = max(1.0, math.hypot(bx - ax, by - ay))
    return out


def apply_rig_constraints(
    nodes: Dict[str, Tuple[float, float]],
    lengths: Dict[Tuple[str, str], float],
    bones: List[Tuple[str, str]],
    locked_nodes: Optional[set] = None,
    iters: int = 6,
) -> Dict[str, Tuple[float, float]]:
    locked_nodes = locked_nodes or set()
    work = {k: (float(v[0]), float(v[1])) for k, v in nodes.items()}
    for _ in range(iters):
        for a, b in bones:
            if a not in work or b not in work:
                continue
            ax, ay = work[a]
            bx, by = work[b]
            dx = bx - ax
            dy = by - ay
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            target = lengths.get((a, b), d)
            corr = (d - target) / d
            mx = dx * 0.5 * corr
            my = dy * 0.5 * corr
            if a in locked_nodes and b in locked_nodes:
                continue
            if a in locked_nodes:
                work[b] = (bx - 2.0 * mx, by - 2.0 * my)
            elif b in locked_nodes:
                work[a] = (ax + 2.0 * mx, ay + 2.0 * my)
            else:
                work[a] = (ax + mx, ay + my)
                work[b] = (bx - mx, by - my)
    return work


def rig_to_pose_map(nodes: Dict[str, Tuple[float, float]], size: Tuple[int, int], bones: List[Tuple[str, str]]) -> Image.Image:
    w, h = size
    pose = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(pose)
    lw = max(2, int(min(w, h) * 0.012))
    for a, b in bones:
        if a not in nodes or b not in nodes:
            continue
        ax, ay = nodes[a]
        bx, by = nodes[b]
        if a.startswith("l_") or b.startswith("l_"):
            c = RIG_COLORS["left"]
        elif a.startswith("r_") or b.startswith("r_"):
            c = RIG_COLORS["right"]
        else:
            c = RIG_COLORS["center"]
        draw.line([(ax, ay), (bx, by)], fill=c, width=lw)
    for n, (x, y) in nodes.items():
        if n.startswith("l_"):
            c = RIG_COLORS["left"]
        elif n.startswith("r_"):
            c = RIG_COLORS["right"]
        else:
            c = RIG_COLORS["center"]
        r = max(2, lw // 2 + 1)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c)
    return pose


def shift_rig_nodes(nodes: Dict[str, Tuple[float, float]], dx: float, dy: float, size: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
    w, h = size
    out = {}
    for k, (x, y) in nodes.items():
        nx = max(0.0, min(float(w - 1), x + dx))
        ny = max(0.0, min(float(h - 1), y + dy))
        out[k] = (nx, ny)
    return out


def scale_rig_nodes(nodes: Dict[str, Tuple[float, float]], old_size: Tuple[int, int], new_size: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
    ow, oh = old_size
    nw, nh = new_size
    sx = nw / max(1, ow)
    sy = nh / max(1, oh)
    return {k: (x * sx, y * sy) for k, (x, y) in nodes.items()}


def apply_chroma_key(img: Image.Image, key_rgb: tuple, tol: int) -> Image.Image:
    rgba = img.convert("RGBA")
    data = np.array(rgba)
    r = data[..., 0]
    g = data[..., 1]
    b = data[..., 2]
    kr, kg, kb = key_rgb
    mask = (
        (np.abs(r - kr) <= tol)
        & (np.abs(g - kg) <= tol)
        & (np.abs(b - kb) <= tol)
    )
    alpha = data[..., 3]
    if mask.shape != alpha.shape and mask.T.shape == alpha.shape:
        mask = mask.T
    data[..., 3] = np.where(mask, 0, alpha)
    return Image.fromarray(data)


def recenter_on_canvas(img: Image.Image, target_w: int, target_h: int, dx: int, dy: int) -> Image.Image:
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    x = (target_w - img.width) // 2 + dx
    y = (target_h - img.height) // 2 + dy
    canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
    return canvas


def crop_image(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    return img.crop((x, y, x + w, y + h))


def shift_image_on_canvas(img: Image.Image, dx: int, dy: int) -> Image.Image:
    rgba = img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    canvas.paste(rgba, (dx, dy), rgba)
    return canvas


def compose_with_offset(img: Image.Image, offset: Tuple[int, int]) -> Image.Image:
    dx, dy = offset
    return shift_image_on_canvas(img, dx, dy)


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


@lru_cache(maxsize=1)
def load_openpose() -> OpenposeDetector:
    return OpenposeDetector.from_pretrained("lllyasviel/Annotators")


@lru_cache(maxsize=4)
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


@lru_cache(maxsize=1)
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
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def pil_to_inline_data(img: Image.Image) -> Dict[str, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"mimeType": "image/png", "data": b64}


def gemini_describe_image(api_key: str, model: str, img: Image.Image, hint: str = "") -> str:
    prompt = (
        "Descrivi in modo preciso il soggetto principale, stile, colori e inquadratura. "
        "Non inventare elementi non presenti. Rispondi con 1-3 frasi concise."
    )
    if hint.strip():
        prompt += f" Indicazioni utente da rispettare: {hint.strip()}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": pil_to_inline_data(img)},
                ]
            }
        ]
    }
    res = requests.post(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        json=body,
        timeout=60,
    )
    if not res.ok:
        raise RuntimeError(f"Gemini error {res.status_code}: {res.text[:240]}")
    payload = res.json()
    parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()
    if not text:
        raise RuntimeError("Gemini non ha restituito testo.")
    return text


def gemini_generate_rig_animation(
    api_key: str,
    model: str,
    base_nodes: Dict[str, Tuple[float, float]],
    bones: List[Tuple[str, str]],
    frame_count: int,
    prompt: str,
    key_descriptions: Dict[int, str],
    key_nodes: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
) -> List[Dict[str, Tuple[float, float]]]:
    # Ask Gemini for normalized coordinates per frame (0..1)
    node_list = list(base_nodes.keys())
    skeleton_hint = "; ".join([f"{a}-{b}" for a, b in bones])
    key_hint = ", ".join([f"{k+1}:{v}" for k, v in key_descriptions.items() if v])
    key_nodes_hint = ""
    if key_nodes:
        try:
            key_nodes_hint = json.dumps(key_nodes, ensure_ascii=True)
        except Exception:
            key_nodes_hint = ""
    example = (
        '{\n'
        '  "frames": [\n'
        '    {"nodes": {"hip": [0.5, 0.6], "neck": [0.5, 0.35]}},\n'
        '    {"nodes": {"hip": [0.5, 0.6], "neck": [0.5, 0.33]}}\n'
        '  ]\n'
        '}\n'
    )
    instruction = (
        "Genera una animazione di rig per sprite 2D.\n"
        "Output richiesto: JSON con campo 'frames' (lista lunga esattamente N),\n"
        "ogni frame ha 'nodes' con coordinate normalizzate 0..1 per ciascun nodo.\n"
        "Non inventare nodi extra. Usa solo quelli forniti.\n"
        "Mantieni coerenza e movimento fluido.\n"
        f"N={frame_count}\n"
        f"Nodi: {', '.join(node_list)}\n"
        f"Segmenti: {skeleton_hint}\n"
        f"Prompt animazione: {prompt}\n"
        f"Descrizioni keyframe (opzionale): {key_hint}\n"
        f"Keyframe nodes (vincoli, opzionale): {key_nodes_hint}\n"
        "Se i keyframe nodes sono forniti, i frame corrispondenti devono rispettare quei valori.\n"
        "Rispondi SOLO con un oggetto JSON. Non usare blocchi di codice, non usare markdown.\n"
        "Se non puoi rispondere, restituisci: {\"frames\": []}\n"
        "Rispondi SOLO con JSON valido, senza testo extra o markdown.\n"
        "Esempio formato:\n"
        f"{example}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": instruction}]}],
        "generationConfig": {"temperature": 0.2},
    }

    last_err = None
    text = ""
    for attempt in range(3):
        try:
            res = requests.post(
                url,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json=body,
                timeout=120,
            )
            if not res.ok:
                raise RuntimeError(f"Gemini error {res.status_code}: {res.text[:240]}")
            payload = res.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()
            if text:
                break
        except Exception as e:
            last_err = e
    if not text:
        raise RuntimeError(f"Nessuna risposta valida da Gemini. Ultimo errore: {last_err}")
    if not text:
        raise RuntimeError("Gemini non ha restituito testo.")
    # Try to parse JSON; if model wrapped it, extract the first JSON object.
    def _extract_json_block(raw: str) -> Optional[str]:
        import re
        if not raw:
            return None
        m = re.search(r"```json\\s*(\\{[\\s\\S]*?\\})\\s*```", raw)
        if m:
            return m.group(1)
        m = re.search(r"\\{[\\s\\S]*\\}", raw)
        if m:
            return m.group(0)
        return None

    def _repair_json_text(raw: str) -> str:
        # Basic repairs: replace fancy quotes, remove trailing commas.
        import re
        s = raw.replace("“", "\"").replace("”", "\"").replace("’", "'")
        s = re.sub(r",\\s*([}\\]])", r"\\1", s)
        return s

    try:
        data = json.loads(text)
    except Exception:
        block = _extract_json_block(text)
        if not block:
            raise RuntimeError("JSON non valido: nessun oggetto JSON trovato.")
        try:
            data = json.loads(block)
        except Exception:
            block = _repair_json_text(block)
            try:
                data = json.loads(block)
            except Exception as e:
                raise RuntimeError(f"JSON non valido: {e}")
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise RuntimeError("Output JSON non valido: frames mancante o lunghezza errata.")
    out: List[Dict[str, Tuple[float, float]]] = []
    for f in frames:
        nodes = f.get("nodes", {}) if isinstance(f, dict) else {}
        frame_nodes: Dict[str, Tuple[float, float]] = {}
        for n in node_list:
            v = nodes.get(n)
            if isinstance(v, list) and len(v) == 2:
                frame_nodes[n] = (float(v[0]), float(v[1]))
        out.append(frame_nodes)
    return out


def preprocess_for_caption(img: Image.Image) -> Image.Image:
    # Pixel-art friendly preprocessing for better captioning.
    src = img.convert("RGB")
    w, h = src.size
    scale = max(1, int(math.ceil(256 / max(1, min(w, h)))))
    up = src.resize((w * scale, h * scale), Image.NEAREST)
    up = ImageEnhance.Contrast(up).enhance(1.25)
    up = up.filter(ImageFilter.SHARPEN)
    return up


def merge_caption_with_hint(auto_caption: str, hint: str) -> str:
    a = (auto_caption or "").strip()
    h = (hint or "").strip()
    if h and a:
        return f"{h}. Dettagli rilevati: {a}"
    if h:
        return h
    return a


class ThumbButton(QPushButton):
    def __init__(self, index: int, parent: "MainWindow"):
        super().__init__(str(index + 1))
        self.index = index
        self.parent_win = parent
        self.setFixedSize(84, 84)
        self.setAcceptDrops(True)
        self._drag_start_pos: Optional[QPoint] = None
        self.clicked.connect(self.on_click)

    def on_click(self):
        self.parent_win.set_selected_frame(self.index)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.parent_win.set_selected_frame(self.index)
            self.parent_win.prompt_upload_for_frame(self.index)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return super().mouseMoveEvent(event)
        if self._drag_start_pos is None:
            return super().mouseMoveEvent(event)
        dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
        if dist < 8:
            return super().mouseMoveEvent(event)
        fr = self.parent_win.frames[self.index]
        if fr.image is None and fr.generated is None:
            return super().mouseMoveEvent(event)

        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(f"frame:{self.index}")
        drag.setMimeData(mime)
        drag.exec(Qt.MoveAction | Qt.CopyAction)

    def dragEnterEvent(self, event):
        text = event.mimeData().text() if event.mimeData() else ""
        if text.startswith("frame:"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        text = event.mimeData().text() if event.mimeData() else ""
        if not text.startswith("frame:"):
            event.ignore()
            return
        try:
            src_idx = int(text.split(":", 1)[1])
        except Exception:
            event.ignore()
            return
        self.parent_win.handle_frame_drop(src_idx, self.index, event.position().toPoint(), self)
        event.acceptProposedAction()

    def update_thumbnail(self, frame: FrameData, selected: bool):
        self.setStyleSheet(
            "QPushButton { border: 2px solid #16a34a; background: #e2fbe8; color: #0f172a; font-weight: 600; }"
            if selected
            else "QPushButton { border: 1px solid #94a3b8; background: #f8fafc; color: #0f172a; }"
        )
        img = frame.generated or (compose_with_offset(frame.image, frame.image_offset) if frame.image is not None else None)
        if img is None:
            self.setIcon(QPixmap())
            self.setText(str(self.index + 1))
            return
        pix = pil_to_qpixmap(img)
        self.setIcon(pix)
        self.setIconSize(QSize(80, 80))
        self.setText("")


class VectorPreviewWidget(QWidget):
    def __init__(self, parent: "MainWindow"):
        super().__init__()
        self.parent_win = parent
        self.setMinimumSize(520, 420)
        self.drag_node: Optional[str] = None
        self._img_rect: Optional[QRect] = None

    def _compose_image(self) -> Optional[Image.Image]:
        fr = self.parent_win.frames[self.parent_win.selected_index]
        if fr.image is None and fr.generated is None:
            # Rig-only frame: use a synthetic canvas based on frame 1 size (if available).
            ref = self.parent_win.frames[0].image
            if ref is not None:
                return Image.new("RGBA", ref.size, (255, 255, 255, 0))
            return None
        if self.parent_win.chk_sprite.isChecked():
            # During eyedropper mode always show/edit the base bitmap layer.
            if self.parent_win.eyedropper_mode and fr.image is not None:
                base_src = compose_with_offset(fr.image, fr.image_offset)
            else:
                base_src = fr.generated or (compose_with_offset(fr.image, fr.image_offset) if fr.image is not None else None)
            if base_src is None:
                ref = self.parent_win.frames[0].image
                if ref is not None:
                    return Image.new("RGBA", ref.size, (255, 255, 255, 0))
                return None
            base = base_src.convert("RGBA")
        else:
            src = fr.generated or fr.image
            if src is None:
                ref = self.parent_win.frames[0].image
                if ref is None:
                    return None
                base = Image.new("RGBA", ref.size, (0, 0, 0, 0))
            else:
                base = Image.new("RGBA", src.size, (0, 0, 0, 0))

        if self.parent_win.chk_onion.isChecked():
            r = self.parent_win.onion_range.value()
            for k in range(max(0, self.parent_win.selected_index - r), min(len(self.parent_win.frames), self.parent_win.selected_index + r + 1)):
                if k == self.parent_win.selected_index:
                    continue
                other = self.parent_win.frames[k].generated or self.parent_win.frames[k].image
                if other is None:
                    continue
                if self.parent_win.frames[k].generated is None:
                    other = compose_with_offset(other, self.parent_win.frames[k].image_offset)
                ov = other.convert("RGBA").copy()
                ov.putalpha(60)
                if ov.size == base.size:
                    base = Image.alpha_composite(base, ov)
        return base

    def _fit_rect(self, img_w: int, img_h: int) -> QRect:
        w = max(1, self.width() - 20)
        h = max(1, self.height() - 20)
        s = min(w / img_w, h / img_h)
        rw = int(img_w * s)
        rh = int(img_h * s)
        x = (self.width() - rw) // 2
        y = (self.height() - rh) // 2
        return QRect(x, y, rw, rh)

    def _widget_to_image(self, p: QPoint, rect: QRect, img_size: Tuple[int, int]) -> Tuple[float, float]:
        iw, ih = img_size
        x = (p.x() - rect.x()) * iw / max(1, rect.width())
        y = (p.y() - rect.y()) * ih / max(1, rect.height())
        return float(x), float(y)

    def _image_to_widget(self, x: float, y: float, rect: QRect, img_size: Tuple[int, int]) -> Tuple[int, int]:
        iw, ih = img_size
        wx = rect.x() + int(x * rect.width() / max(1, iw))
        wy = rect.y() + int(y * rect.height() / max(1, ih))
        return wx, wy

    def _closest_node(self, pt: QPoint, rect: QRect, fr: FrameData, threshold: int = 12) -> Optional[str]:
        if fr.rig_nodes is None:
            return None
        canvas_size = self.parent_win.get_frame_canvas_size(self.parent_win.selected_index)
        if canvas_size is None:
            return None
        best = None
        best_d = threshold
        for n, (x, y) in fr.rig_nodes.items():
            wx, wy = self._image_to_widget(x, y, rect, canvas_size)
            d = math.hypot(pt.x() - wx, pt.y() - wy)
            if d <= best_d:
                best = n
                best_d = d
        return best

    def _closest_bone(self, pt: QPoint, rect: QRect, fr: FrameData, threshold: float = 10.0) -> Optional[Tuple[str, str]]:
        if fr.rig_nodes is None:
            return None
        canvas_size = self.parent_win.get_frame_canvas_size(self.parent_win.selected_index)
        if canvas_size is None:
            return None
        best_bone = None
        best_d = threshold
        for a, b in self._frame_bones(fr):
            if a not in fr.rig_nodes or b not in fr.rig_nodes:
                continue
            ax, ay = self._image_to_widget(fr.rig_nodes[a][0], fr.rig_nodes[a][1], rect, canvas_size)
            bx, by = self._image_to_widget(fr.rig_nodes[b][0], fr.rig_nodes[b][1], rect, canvas_size)
            px, py = pt.x(), pt.y()
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            c1 = vx * wx + vy * wy
            c2 = vx * vx + vy * vy
            if c2 <= 1e-6:
                continue
            t = max(0.0, min(1.0, c1 / c2))
            projx = ax + t * vx
            projy = ay + t * vy
            d = math.hypot(px - projx, py - projy)
            if d <= best_d:
                best_d = d
                best_bone = (a, b)
        return best_bone

    def _closest_other_node(self, pt: QPoint, rect: QRect, fr: FrameData, exclude: str, threshold: float = 12.0) -> Optional[str]:
        if fr.rig_nodes is None:
            return None
        canvas_size = self.parent_win.get_frame_canvas_size(self.parent_win.selected_index)
        if canvas_size is None:
            return None
        best = None
        best_d = threshold
        for n, (x, y) in fr.rig_nodes.items():
            if n == exclude:
                continue
            wx, wy = self._image_to_widget(x, y, rect, canvas_size)
            d = math.hypot(pt.x() - wx, pt.y() - wy)
            if d <= best_d:
                best = n
                best_d = d
        return best

    def _frame_bones(self, fr: FrameData) -> List[Tuple[str, str]]:
        profile = self.parent_win.resolve_rig_profile(fr)
        return self.parent_win.get_frame_bones(fr)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setPen(QPen(QColor("#94a3b8"), 2))
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))

        img = self._compose_image()
        if img is None:
            painter.setPen(QPen(QColor("#475569")))
            painter.drawText(self.rect(), Qt.AlignCenter, "Nessun frame")
            return

        rect = self._fit_rect(img.width, img.height)
        self._img_rect = rect
        pix = pil_to_qpixmap(img)
        painter.drawPixmap(rect, pix)

        fr = self.parent_win.frames[self.parent_win.selected_index]
        if self.parent_win.chk_pose.isChecked() and fr.rig_nodes is not None:
            bones = self._frame_bones(fr)
            used_nodes = set()
            for a, b in bones:
                if a not in fr.rig_nodes or b not in fr.rig_nodes:
                    continue
                used_nodes.add(a)
                used_nodes.add(b)
                ax, ay = fr.rig_nodes[a]
                bx, by = fr.rig_nodes[b]
                awx, awy = self._image_to_widget(ax, ay, rect, img.size)
                bwx, bwy = self._image_to_widget(bx, by, rect, img.size)
                if a.startswith("l_") or b.startswith("l_"):
                    c = QColor(*RIG_COLORS["left"])
                elif a.startswith("r_") or b.startswith("r_"):
                    c = QColor(*RIG_COLORS["right"])
                else:
                    c = QColor(*RIG_COLORS["center"])
                painter.setPen(QPen(c, 3))
                painter.drawLine(awx, awy, bwx, bwy)

            for n, (x, y) in fr.rig_nodes.items():
                if n not in used_nodes:
                    continue
                wx, wy = self._image_to_widget(x, y, rect, img.size)
                if n.startswith("l_"):
                    c = QColor(*RIG_COLORS["left"])
                elif n.startswith("r_"):
                    c = QColor(*RIG_COLORS["right"])
                else:
                    c = QColor(*RIG_COLORS["center"])
                # Draw a small background halo to visually break crossing segments at joints.
                painter.setPen(QPen(QColor("#ffffff"), 0))
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(wx - 7, wy - 7, 14, 14)
                painter.setPen(QPen(QColor("#0f172a"), 1))
                painter.setBrush(QBrush(c))
                painter.drawEllipse(wx - 5, wy - 5, 10, 10)

    def mousePressEvent(self, event):
        fr = self.parent_win.frames[self.parent_win.selected_index]
        if fr.rig_nodes is None or self._img_rect is None:
            return
        if event.button() == Qt.LeftButton and self.parent_win.eyedropper_mode:
            img = fr.generated or (compose_with_offset(fr.image, fr.image_offset) if fr.image is not None else None)
            if img is not None:
                nx, ny = self._widget_to_image(event.position().toPoint(), self._img_rect, img.size)
                self.parent_win.apply_eyedropper_pixel(int(nx), int(ny))
            return
        if event.button() == Qt.RightButton:
            if self.parent_win.selected_index != 0:
                self.parent_win.log("Topologia modificabile solo nel frame 1")
                return
            hit_node = self._closest_node(event.position().toPoint(), self._img_rect, fr)
            if hit_node is not None:
                self.parent_win.delete_topology_node(hit_node)
                return
            hit_bone = self._closest_bone(event.position().toPoint(), self._img_rect, fr)
            if hit_bone is not None:
                self.parent_win.split_topology_bone(hit_bone[0], hit_bone[1])
                return
        self.drag_node = self._closest_node(event.position().toPoint(), self._img_rect, fr)

    def mouseMoveEvent(self, event):
        if self.drag_node is None:
            return
        fr = self.parent_win.frames[self.parent_win.selected_index]
        canvas_size = self.parent_win.get_frame_canvas_size(self.parent_win.selected_index)
        if canvas_size is None or fr.rig_nodes is None or fr.bone_lengths is None or self._img_rect is None:
            return
        nx, ny = self._widget_to_image(event.position().toPoint(), self._img_rect, canvas_size)
        nx = max(0.0, min(float(canvas_size[0] - 1), nx))
        ny = max(0.0, min(float(canvas_size[1] - 1), ny))
        fr.rig_nodes[self.drag_node] = (nx, ny)
        free_edit = bool(event.modifiers() & Qt.ShiftModifier)
        if not free_edit:
            fr.rig_nodes = apply_rig_constraints(
                fr.rig_nodes,
                fr.bone_lengths,
                self._frame_bones(fr),
                locked_nodes={self.drag_node},
                iters=5,
            )
        else:
            # Shift-drag: move only this node and accept new segment lengths.
            fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, self._frame_bones(fr))
        self.parent_win.frames[self.parent_win.selected_index] = fr
        self.parent_win.mark_rig_dirty()
        self.update()

    def mouseReleaseEvent(self, _event):
        if self.drag_node is not None and self._img_rect is not None:
            fr = self.parent_win.frames[self.parent_win.selected_index]
            if fr.rig_nodes is not None:
                # Shift-release over another node -> merge/weld nodes.
                if bool(_event.modifiers() & Qt.ShiftModifier):
                    if self.parent_win.selected_index == 0:
                        target = self._closest_other_node(_event.position().toPoint(), self._img_rect, fr, self.drag_node, threshold=14.0)
                        if target is not None:
                            self.parent_win.merge_topology_nodes(self.drag_node, target)
                    else:
                        self.parent_win.log("Merge nodi consentito solo nel frame 1")
        self.drag_node = None

class DetailDialog(QDialog):
    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("Dettaglio immagine")
        self.setMinimumWidth(320)

        layout = QFormLayout(self)

        self.target_w = QSpinBox()
        self.target_w.setRange(16, 4096)
        self.target_h = QSpinBox()
        self.target_h.setRange(16, 4096)

        self.brightness = QDoubleSpinBox()
        self.brightness.setRange(0.5, 2.0)
        self.brightness.setSingleStep(0.1)
        self.brightness.setValue(1.0)

        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.5, 2.0)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1.0)

        self.saturation = QDoubleSpinBox()
        self.saturation.setRange(0.5, 2.0)
        self.saturation.setSingleStep(0.1)
        self.saturation.setValue(1.0)

        self.key_color = QLineEdit("#00ff00")
        self.key_tol = QSpinBox()
        self.key_tol.setRange(0, 80)
        self.key_tol.setValue(10)
        self.key_enabled = QCheckBox("Attiva trasparenza colore")
        self.key_enabled.setChecked(False)
        self.res_label = QLabel("Risoluzione: -")
        self.key_desc_label = QLabel("Descrizione keyframe")
        self.key_desc_view = QTextEdit()
        self.key_desc_view.setReadOnly(True)
        self.key_desc_view.setFixedHeight(84)

        layout.addRow("Target W", self.target_w)
        layout.addRow("Target H", self.target_h)
        layout.addRow("Brightness", self.brightness)
        layout.addRow("Contrast", self.contrast)
        layout.addRow("Saturation", self.saturation)
        layout.addRow(self.key_enabled)
        layout.addRow("Key color", self.key_color)
        layout.addRow("Key tolerance", self.key_tol)
        layout.addRow(self.res_label)
        layout.addRow(self.key_desc_label)
        layout.addRow(self.key_desc_view)

        btn_apply = QPushButton("Applica")
        btn_apply.clicked.connect(self.apply)
        layout.addRow(btn_apply)

    def sync_from_frame(self, fr: FrameData):
        if fr.image is None:
            self.res_label.setText("Risoluzione: -")
            self.key_desc_label.setVisible(False)
            self.key_desc_view.setVisible(False)
            return
        self.target_w.setValue(fr.image.width)
        self.target_h.setValue(fr.image.height)
        self.res_label.setText(f"Risoluzione: {fr.image.width} x {fr.image.height}")
        if fr.is_keyframe:
            self.key_desc_label.setVisible(True)
            self.key_desc_view.setVisible(True)
            self.key_desc_view.setPlainText(fr.key_description)
        else:
            self.key_desc_label.setVisible(False)
            self.key_desc_view.setVisible(False)

    def apply(self):
        self.parent_win.apply_detail_changes(
            self.target_w.value(),
            self.target_h.value(),
            self.brightness.value(),
            self.contrast.value(),
            self.saturation.value(),
            self.key_enabled.isChecked(),
            self.key_color.text().strip(),
            self.key_tol.value(),
        )
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sprite Animator")
        self.resize(1400, 900)

        self.frames: List[FrameData] = [FrameData() for _ in range(8)]
        self.selected_index = 0
        self.thumbs: List[ThumbButton] = []
        self.eyedropper_mode: bool = False
        self._pose_visible_before_eyedropper: bool = True
        self.last_eyedrop_color: Tuple[int, int, int] = (0, 0, 0)
        self.rig_approved: bool = False
        self.topology_bones: Optional[List[Tuple[str, str]]] = None
        self.topology_new_node_counter: int = 1

        self.detail_dialog = DetailDialog(self)

        self.build_ui()
        self.refresh_ui()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        top = QHBoxLayout()

        # Left options panel (scrollable)
        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)

        title = QLabel("opzioni")
        title.setObjectName("title")
        left_layout.addWidget(title)

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("prompt")
        self.negative = QTextEdit()
        self.negative.setPlaceholderText("prompt negative")
        left_layout.addWidget(self.prompt)
        left_layout.addWidget(self.negative)

        self.key_desc_label = QLabel("Descrizione frame chiave")
        self.key_desc = QTextEdit()
        self.key_desc.setPlaceholderText("Inserisci descrizione solo per questo keyframe")
        self.key_desc.textChanged.connect(self.on_key_desc_changed)
        left_layout.addWidget(self.key_desc_label)
        left_layout.addWidget(self.key_desc)

        self.caption_hint_label = QLabel("Hint descrizione (opzionale)")
        self.caption_hint = QTextEdit()
        self.caption_hint.setFixedHeight(64)
        self.caption_hint.setPlaceholderText("Es: guerriero umano pixel art, capelli biondi, barba")
        self.gemini_api_key = QLineEdit()
        self.gemini_api_key.setEchoMode(QLineEdit.Password)
        self.gemini_api_key.setPlaceholderText("API key Gemini (opzionale)")
        self.gemini_model = QLineEdit("gemini-2.5-flash")
        self.use_gemini_caption = QCheckBox("Usa Gemini per descrizione")
        self.use_gemini_caption.setChecked(True)
        self.btn_recalc_caption = QPushButton("Ricalcola descrizione")
        self.btn_recalc_caption.clicked.connect(self.recalculate_caption_selected)
        self.caption_current_label = QLabel("Descrizione frame: -")
        self.caption_current_label.setWordWrap(True)
        left_layout.addWidget(self.caption_hint_label)
        left_layout.addWidget(self.caption_hint)
        left_layout.addWidget(self.gemini_api_key)
        left_layout.addWidget(self.gemini_model)
        left_layout.addWidget(self.use_gemini_caption)
        left_layout.addWidget(self.btn_recalc_caption)
        left_layout.addWidget(self.caption_current_label)

        self.rig_profile_label = QLabel("Tipo rig frame")
        self.rig_profile_combo = QComboBox()
        self.rig_profile_combo.addItems(["auto", "human", "animal", "object"])
        self.rig_profile_combo.currentTextChanged.connect(self.on_rig_profile_changed)
        self.rig_detected_label = QLabel("Rilevato: -")
        left_layout.addWidget(self.rig_profile_label)
        left_layout.addWidget(self.rig_profile_combo)
        left_layout.addWidget(self.rig_detected_label)

        self.btn_generate_pose = QPushButton("simula rig")
        self.btn_generate_pose.clicked.connect(self.simulate_rig_animation)
        self.btn_ai_rig = QPushButton("AI rig")
        self.btn_ai_rig.clicked.connect(self.generate_rig_with_ai)
        self.btn_approve_rig = QPushButton("approva rig")
        self.btn_approve_rig.setCheckable(True)
        self.btn_approve_rig.toggled.connect(self.set_rig_approved)
        self.btn_generate_frames = QPushButton("genera raster")
        self.btn_generate_frames.clicked.connect(self.generate_final_frames)
        self.btn_generate_frames.setEnabled(False)

        left_layout.addWidget(self.btn_generate_pose)
        left_layout.addWidget(self.btn_ai_rig)
        left_layout.addWidget(self.btn_approve_rig)
        left_layout.addWidget(self.btn_generate_frames)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box)

        left_layout.addStretch(1)
        left_panel.setWidget(left_inner)

        # Right detail panel
        right_panel = QVBoxLayout()
        right_top = QHBoxLayout()
        self.btn_detail = QPushButton("dettaglio")
        self.btn_detail.clicked.connect(self.open_detail)
        right_top.addWidget(self.btn_detail)
        right_top.addStretch(1)
        self.chk_sprite = QCheckBox("raster")
        self.chk_sprite.setChecked(True)
        self.chk_sprite.toggled.connect(self.refresh_preview)
        self.chk_pose = QCheckBox("scheletro")
        self.chk_pose.setChecked(True)
        self.chk_pose.toggled.connect(self.refresh_preview)
        right_top.addWidget(self.chk_sprite)
        right_top.addWidget(self.chk_pose)
        self.btn_eyedropper = QPushButton("contagocce")
        self.btn_eyedropper.setCheckable(True)
        self.btn_eyedropper.toggled.connect(self.toggle_eyedropper_mode)
        right_top.addWidget(self.btn_eyedropper)

        self.preview = VectorPreviewWidget(self)
        self.preview.setObjectName("preview")

        preview_shell = QWidget()
        preview_grid = QGridLayout(preview_shell)
        preview_grid.setContentsMargins(0, 0, 0, 0)
        preview_grid.setHorizontalSpacing(8)
        preview_grid.setVerticalSpacing(8)

        self.btn_move_left = QPushButton("←")
        self.btn_move_up = QPushButton("↑")
        self.btn_move_down = QPushButton("↓")
        self.btn_move_right = QPushButton("→")
        self.btn_move_left.setFixedSize(32, 32)
        self.btn_move_up.setFixedSize(32, 32)
        self.btn_move_down.setFixedSize(32, 32)
        self.btn_move_right.setFixedSize(32, 32)

        self.btn_move_left.clicked.connect(lambda: self.move_selected_image(-self.preview_move_step.value(), 0))
        self.btn_move_up.clicked.connect(lambda: self.move_selected_image(0, -self.preview_move_step.value()))
        self.btn_move_down.clicked.connect(lambda: self.move_selected_image(0, self.preview_move_step.value()))
        self.btn_move_right.clicked.connect(lambda: self.move_selected_image(self.preview_move_step.value(), 0))

        preview_grid.addWidget(self.btn_move_up, 0, 1, alignment=Qt.AlignCenter)
        preview_grid.addWidget(self.btn_move_left, 1, 0, alignment=Qt.AlignCenter)
        preview_grid.addWidget(self.preview, 1, 1, alignment=Qt.AlignCenter)
        preview_grid.addWidget(self.btn_move_right, 1, 2, alignment=Qt.AlignCenter)
        preview_grid.addWidget(self.btn_move_down, 2, 1, alignment=Qt.AlignCenter)

        right_bottom = QHBoxLayout()
        self.chk_onion = QCheckBox("onion skin")
        self.onion_range = QSpinBox()
        self.onion_range.setRange(1, 10)
        self.onion_range.setValue(2)
        self.preview_move_step = QSpinBox()
        self.preview_move_step.setRange(1, 64)
        self.preview_move_step.setValue(2)
        self.eyedropper_tol = QSpinBox()
        self.eyedropper_tol.setRange(0, 80)
        self.eyedropper_tol.setValue(10)
        self.eyedropper_color_chip = QLabel()
        self.eyedropper_color_chip.setFixedSize(22, 22)
        self.eyedropper_color_chip.setToolTip("Ultimo colore selezionato")
        self._update_eyedropper_chip()
        right_bottom.addWidget(self.chk_onion)
        right_bottom.addWidget(QLabel("range"))
        right_bottom.addWidget(self.onion_range)
        right_bottom.addWidget(QLabel("step"))
        right_bottom.addWidget(self.preview_move_step)
        right_bottom.addWidget(QLabel("tol"))
        right_bottom.addWidget(self.eyedropper_tol)
        right_bottom.addWidget(QLabel("col"))
        right_bottom.addWidget(self.eyedropper_color_chip)
        right_bottom.addStretch(1)

        right_panel.addLayout(right_top)
        right_panel.addWidget(preview_shell, alignment=Qt.AlignCenter)
        right_panel.addLayout(right_bottom)

        top.addWidget(left_panel, 1)
        top.addLayout(right_panel, 2)

        # Bottom timeline panel
        bottom = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_prev = QPushButton("<")
        self.btn_next = QPushButton(">")
        self.btn_prev.clicked.connect(lambda: self.step_frame(-1))
        self.btn_next.clicked.connect(lambda: self.step_frame(1))

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        thumb_container = QWidget()
        self.thumb_layout = QHBoxLayout(thumb_container)
        self.thumb_layout.setContentsMargins(8, 8, 8, 8)
        self.thumb_layout.setSpacing(8)
        self.thumb_scroll.setWidget(thumb_container)

        row.addWidget(self.btn_prev)
        row.addWidget(self.thumb_scroll, 1)
        row.addWidget(self.btn_next)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(2, 24)
        self.frame_slider.setValue(8)
        self.frame_slider.valueChanged.connect(self.on_frame_count_changed)

        self.frame_label = QLabel("FRAME 1/8")
        self.frame_label.setAlignment(Qt.AlignCenter)

        bottom.addLayout(row)
        bottom.addWidget(self.frame_label)
        bottom.addWidget(self.frame_slider)

        main.addLayout(top, 3)
        main.addLayout(bottom, 1)

        self.apply_style()

    def apply_style(self):
        self.setStyleSheet(
            "QMainWindow { background: #f3f4f6; color: #0f172a; }"
            "QScrollArea { background: #f3f4f6; border: none; }"
            "QWidget { font-family: 'Segoe UI'; font-size: 12px; color: #0f172a; }"
            "QLabel { color: #0f172a; }"
            "QTextEdit, QLineEdit { background: #ffffff; border: 1px solid #94a3b8; color: #0f172a; selection-background-color: #bfdbfe; }"
            "QTextEdit:focus, QLineEdit:focus { border: 1px solid #2563eb; }"
            "QPushButton { background: #e2e8f0; color: #0f172a; border: 1px solid #94a3b8; padding: 6px 10px; }"
            "QPushButton:hover { background: #cbd5e1; }"
            "QCheckBox { color: #0f172a; spacing: 6px; }"
            "QSpinBox, QDoubleSpinBox, QComboBox { background: #ffffff; color: #0f172a; border: 1px solid #94a3b8; }"
            "QLabel#title { font-size: 48px; font-weight: 700; color: #334155; }"
            "QWidget#preview { background: #ffffff; border: 2px solid #94a3b8; }"
            "QSlider::groove:horizontal { height: 6px; background: #cbd5e1; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -5px 0; border-radius: 7px; }"
        )

    def log(self, msg: str):
        self.log_box.append(msg)

    def refresh_ui(self):
        self.rebuild_thumbs()
        self.refresh_preview()
        self.update_frame_label()
        self.sync_keyframe_editor()
        self.btn_generate_frames.setEnabled(self.rig_approved)
        self.btn_approve_rig.blockSignals(True)
        self.btn_approve_rig.setChecked(self.rig_approved)
        self.btn_approve_rig.blockSignals(False)

    def mark_rig_dirty(self):
        if self.rig_approved:
            self.rig_approved = False
            self.btn_generate_frames.setEnabled(False)
            self.btn_approve_rig.blockSignals(True)
            self.btn_approve_rig.setChecked(False)
            self.btn_approve_rig.blockSignals(False)
            self.log("Rig modificato: approvazione rimossa")

    def set_rig_approved(self, approved: bool):
        self.rig_approved = bool(approved)
        self.btn_generate_frames.setEnabled(self.rig_approved)
        self.log("Rig approvato: puoi generare raster" if self.rig_approved else "Rig non approvato")

    def toggle_eyedropper_mode(self, enabled: bool):
        self.eyedropper_mode = bool(enabled)
        if enabled:
            self.chk_sprite.setChecked(True)
            self._pose_visible_before_eyedropper = self.chk_pose.isChecked()
            self.chk_pose.setChecked(False)
        if self.eyedropper_mode:
            self.preview.setCursor(Qt.CrossCursor)
        else:
            self.preview.unsetCursor()
            self.chk_pose.setChecked(self._pose_visible_before_eyedropper)
        self.log("Contagocce attivo: clicca un colore nella preview" if enabled else "Contagocce disattivato")

    def _update_eyedropper_chip(self):
        r, g, b = self.last_eyedrop_color
        self.eyedropper_color_chip.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid #334155;"
        )

    def apply_eyedropper_pixel(self, x: int, y: int):
        fr = self.frames[self.selected_index]
        if fr.image is None and fr.generated is None:
            return

        x = int(x)
        y = int(y)
        tol = self.eyedropper_tol.value()

        # Always operate on the base bitmap layer.
        if fr.image is None and fr.generated is not None:
            fr.image = fr.generated.convert("RGBA")
            fr.generated = None

        if fr.image is None:
            self.log("Contagocce: nessun bitmap base disponibile")
            return

        ox, oy = fr.image_offset
        sx = x - ox
        sy = y - oy
        if sx < 0 or sy < 0 or sx >= fr.image.width or sy >= fr.image.height:
            self.log("Contagocce: click fuori dal bitmap (offset attivo), nessuna modifica")
            return

        src = fr.image.convert("RGBA")
        r, g, b, a = src.getpixel((sx, sy))
        if a == 0:
            self.log("Contagocce: pixel gia trasparente, nessuna modifica")
            return

        fr.image = apply_chroma_key(fr.image, (r, g, b), tol)
        # Clear generated layer to avoid masking the edited bitmap.
        fr.generated = None
        self.frames[self.selected_index] = fr

        self.detail_dialog.key_enabled.setChecked(True)
        self.detail_dialog.key_color.setText(f"#{r:02x}{g:02x}{b:02x}")
        self.detail_dialog.key_tol.setValue(tol)
        self.last_eyedrop_color = (r, g, b)
        self._update_eyedropper_chip()
        self.log(f"Contagocce: rimosso colore bitmap #{r:02x}{g:02x}{b:02x} (tol={tol})")

        self.frames[self.selected_index] = fr

        self.eyedropper_mode = False
        self.btn_eyedropper.blockSignals(True)
        self.btn_eyedropper.setChecked(False)
        self.btn_eyedropper.blockSignals(False)
        self.refresh_ui()

    def rebuild_thumbs(self):
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.thumbs = []
        for i in range(len(self.frames)):
            frame_box = QWidget()
            frame_box_layout = QVBoxLayout(frame_box)
            frame_box_layout.setContentsMargins(0, 0, 0, 0)
            frame_box_layout.setSpacing(4)

            btn = ThumbButton(i, self)
            self.thumbs.append(btn)
            trash_btn = QPushButton("trash")
            trash_btn.setFixedSize(84, 22)
            trash_btn.setStyleSheet(
                "QPushButton { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; font-weight: 600; }"
                "QPushButton:hover { background: #fecaca; }"
            )
            trash_btn.clicked.connect(lambda _=False, idx=i: self.clear_frame(idx))

            key_cb = QCheckBox("key")
            key_cb.setChecked(self.frames[i].is_keyframe)
            key_cb.toggled.connect(lambda checked, idx=i: self.set_frame_keyframe(idx, checked))

            frame_box_layout.addWidget(btn)
            frame_box_layout.addWidget(key_cb, alignment=Qt.AlignCenter)
            frame_box_layout.addWidget(trash_btn)
            self.thumb_layout.addWidget(frame_box)
        self.thumb_layout.addStretch(1)
        self.refresh_thumbs()

    def refresh_thumbs(self):
        for i, btn in enumerate(self.thumbs):
            btn.update_thumbnail(self.frames[i], i == self.selected_index)

    def update_frame_label(self):
        self.frame_label.setText(f"FRAME {self.selected_index + 1}/{len(self.frames)}")

    def get_frame_canvas_size(self, idx: int) -> Optional[Tuple[int, int]]:
        if idx < 0 or idx >= len(self.frames):
            return None
        fr = self.frames[idx]
        if fr.generated is not None:
            return fr.generated.size
        if fr.image is not None:
            return fr.image.size
        # Rig-only frames inherit canvas from frame 1 if available.
        fr0 = self.frames[0] if self.frames else None
        if fr0 is not None:
            if fr0.generated is not None:
                return fr0.generated.size
            if fr0.image is not None:
                return fr0.image.size
        return None

    def on_frame_count_changed(self, n: int):
        cur = len(self.frames)
        if n > cur:
            self.frames.extend([FrameData() for _ in range(n - cur)])
        elif n < cur:
            self.frames = self.frames[:n]
            self.selected_index = min(self.selected_index, n - 1)
        self.refresh_ui()

    def set_selected_frame(self, idx: int):
        self.selected_index = max(0, min(idx, len(self.frames) - 1))
        self.refresh_thumbs()
        self.refresh_preview()
        self.update_frame_label()
        self.sync_keyframe_editor()
    def handle_frame_drop(self, src_idx: int, dst_idx: int, local_pos: QPoint, source_widget: QWidget):
        if src_idx == dst_idx:
            return
        if src_idx < 0 or src_idx >= len(self.frames) or dst_idx < 0 or dst_idx >= len(self.frames):
            return

        menu = QMenu(source_widget)
        act_move = menu.addAction("sposta")
        act_copy = menu.addAction("duplica")
        act_cancel = menu.addAction("annulla")
        chosen = menu.exec(source_widget.mapToGlobal(local_pos))
        if chosen is None or chosen == act_cancel:
            return

        src_frame = self.frames[src_idx]
        if chosen == act_move:
            self.frames[dst_idx] = src_frame
            self.frames[src_idx] = FrameData()
        elif chosen == act_copy:
            self.frames[dst_idx] = copy.deepcopy(src_frame)
        else:
            return

        self.set_selected_frame(dst_idx)
        self.refresh_ui()

    def prompt_upload_for_frame(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(self, "Carica immagine frame", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        img = Image.open(path).convert("RGB")
        self.set_frame_image(idx, img)

    def clear_frame(self, idx: int):
        self.frames[idx] = FrameData()
        if idx == 0:
            self.topology_bones = None
            self.topology_new_node_counter = 1
        self.mark_rig_dirty()
        if self.selected_index == idx:
            self.refresh_preview()
        self.refresh_ui()
        self.sync_keyframe_editor()

    def set_frame_keyframe(self, idx: int, checked: bool):
        self.frames[idx].is_keyframe = bool(checked)
        if checked and not self.frames[idx].key_description and self.frames[idx].caption:
            self.frames[idx].key_description = self.frames[idx].caption
        if not checked:
            self.frames[idx].key_description = ""
        self.mark_rig_dirty()
        if idx == self.selected_index:
            self.sync_keyframe_editor()

    def sync_keyframe_editor(self):
        fr = self.frames[self.selected_index]
        enabled = fr.is_keyframe
        self.key_desc_label.setEnabled(enabled)
        self.key_desc.setEnabled(enabled)
        self.key_desc.blockSignals(True)
        self.key_desc.setPlainText(fr.key_description if enabled else "")
        self.key_desc.blockSignals(False)
        self.rig_profile_combo.blockSignals(True)
        self.rig_profile_combo.setCurrentText(fr.rig_profile)
        self.rig_profile_combo.blockSignals(False)
        self.rig_detected_label.setText(f"Rilevato: {fr.detected_profile}")
        engine_state = "gemini" if self.use_gemini_caption.isChecked() and self.gemini_api_key.text().strip() else "fallback"
        self.caption_current_label.setText(f"Descrizione frame ({engine_state}): {fr.caption if fr.caption else '-'}")

    def on_key_desc_changed(self):
        fr = self.frames[self.selected_index]
        if not fr.is_keyframe:
            return
        fr.key_description = self.key_desc.toPlainText().strip()
        self.frames[self.selected_index] = fr

    def recalculate_caption_selected(self):
        fr = self.frames[self.selected_index]
        if fr.image is None:
            QMessageBox.information(self, "Descrizione", "Carica prima un'immagine nel frame selezionato.")
            return

        hint = self.caption_hint.toPlainText().strip()
        auto = ""
        engine_used = "none"
        if self.use_gemini_caption.isChecked() and self.gemini_api_key.text().strip():
            try:
                auto = gemini_describe_image(
                    self.gemini_api_key.text().strip(),
                    self.gemini_model.text().strip() or "gemini-2.5-flash",
                    preprocess_for_caption(fr.image),
                    hint=hint,
                )
                engine_used = "gemini"
            except Exception as e:
                self.log(f"[caption] Gemini fallito sul frame {self.selected_index + 1}: {e}")
        elif self.use_gemini_caption.isChecked() and not self.gemini_api_key.text().strip():
            self.log("[caption] Gemini attivo ma API key mancante: uso fallback locale")
        if not auto:
            if not CAPTION_OK:
                self.log("[caption] Nessun motore disponibile (Gemini/fallback)")
                self.caption_current_label.setText("Descrizione frame: errore (nessun motore disponibile)")
                return
            try:
                auto = caption_image(preprocess_for_caption(fr.image))
                engine_used = "fallback"
            except Exception as e:
                self.log(f"[caption] Fallback locale fallito sul frame {self.selected_index + 1}: {e}")
                self.caption_current_label.setText("Descrizione frame: errore (fallback fallito)")
                return

        final_caption = merge_caption_with_hint(auto, hint)
        fr.caption = final_caption
        fr.detected_profile = infer_subject_profile(final_caption)
        if fr.is_keyframe:
            fr.key_description = final_caption
        self.frames[self.selected_index] = fr
        self.sync_keyframe_editor()
        self.log(f"[caption] Descrizione aggiornata frame {self.selected_index + 1} (engine={engine_used})")

    def on_rig_profile_changed(self, value: str):
        fr = self.frames[self.selected_index]
        fr.rig_profile = value
        self.frames[self.selected_index] = fr
        if fr.image is not None:
            if self.selected_index == 0:
                fr.rig_nodes = self.create_default_rig(fr.image, fr)
                self.frames[0] = fr
                self.initialize_topology_from_frame1()
                self.sync_all_frames_to_topology()
            else:
                self._apply_topology_to_frame(self.selected_index)
        self.mark_rig_dirty()
        self.refresh_ui()

    def step_frame(self, delta: int):
        self.set_selected_frame(self.selected_index + delta)

    def open_detail(self):
        fr = self.frames[self.selected_index]
        if fr.image is None:
            QMessageBox.information(self, "Dettaglio", "Carica un'immagine nel frame selezionato.")
            return
        self.detail_dialog.sync_from_frame(fr)
        self.detail_dialog.exec()

    def apply_detail_changes(
        self,
        target_w: int,
        target_h: int,
        brightness: float,
        contrast: float,
        saturation: float,
        key_enabled: bool,
        key_color: str,
        key_tol: int,
    ):
        fr = self.frames[self.selected_index]
        if fr.image is None:
            return

        old_size = fr.image.size
        img = fr.image.convert("RGBA")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)

        if key_enabled and len(key_color) == 7 and key_color.startswith("#"):
            kr = int(key_color[1:3], 16)
            kg = int(key_color[3:5], 16)
            kb = int(key_color[5:7], 16)
            img = apply_chroma_key(img, (kr, kg, kb), key_tol)

        fr.image = img
        fr.image_offset = (0, 0)
        fr.generated = None
        if fr.rig_nodes is None:
            fr.rig_nodes = self.create_default_rig(img, fr)
        else:
            fr.rig_nodes = scale_rig_nodes(fr.rig_nodes, old_size, img.size)
        bones = self.get_frame_bones(fr)
        fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, bones) if fr.rig_nodes is not None else None
        self.frames[self.selected_index] = fr
        if self.selected_index == 0:
            self.sync_all_frames_to_topology()
        self.mark_rig_dirty()
        self.refresh_ui()

    def move_selected_image(self, dx: int, dy: int):
        fr = self.frames[self.selected_index]
        if fr.image is None:
            return
        ox, oy = fr.image_offset
        fr.image_offset = (ox + int(dx), oy + int(dy))
        fr.generated = None
        if fr.rig_nodes is not None:
            fr.rig_nodes = shift_rig_nodes(fr.rig_nodes, dx, dy, fr.image.size)
        else:
            fr.rig_nodes = self.create_default_rig(fr.image, fr)
        bones = self.get_frame_bones(fr)
        fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, bones) if fr.rig_nodes is not None else None
        self.frames[self.selected_index] = fr
        if self.selected_index == 0:
            self.sync_all_frames_to_topology()
        self.mark_rig_dirty()
        self.refresh_ui()

    def resolve_rig_profile(self, fr: FrameData) -> str:
        if fr.rig_profile != "auto":
            return fr.rig_profile
        return fr.detected_profile or "human"

    def get_frame_bones(self, fr: FrameData) -> List[Tuple[str, str]]:
        if self.topology_bones is None and self.frames and self.frames[0].rig_nodes is not None:
            self.initialize_topology_from_frame1()
        if self.topology_bones is not None:
            return self.topology_bones
        return RIG_BONES_BY_PROFILE.get(self.resolve_rig_profile(fr), HUMAN_BONES)

    def _topology_nodes_set(self) -> set:
        if self.topology_bones is None:
            return set()
        s = set()
        for a, b in self.topology_bones:
            s.add(a)
            s.add(b)
        return s

    def initialize_topology_from_frame1(self):
        fr0 = self.frames[0]
        if fr0.rig_nodes is None:
            self.topology_bones = None
            return
        base_bones = RIG_BONES_BY_PROFILE.get(self.resolve_rig_profile(fr0), HUMAN_BONES)
        self.topology_bones = [(a, b) for (a, b) in base_bones if a in fr0.rig_nodes and b in fr0.rig_nodes]
        if not self.topology_bones:
            self.topology_bones = list(base_bones)

    def _apply_topology_to_frame(self, idx: int):
        if self.topology_bones is None:
            return
        fr = self.frames[idx]
        if fr.image is None:
            return
        if fr.rig_nodes is None:
            fr.rig_nodes = self.create_default_rig(fr.image, fr)
        if fr.rig_nodes is None:
            return
        nodes_needed = self._topology_nodes_set()
        # remove nodes no longer part of topology
        for n in list(fr.rig_nodes.keys()):
            if n not in nodes_needed:
                del fr.rig_nodes[n]
        # add missing nodes by inheriting frame 1 normalized position if possible
        fr0 = self.frames[0]
        if fr0.image is not None and fr0.rig_nodes is not None:
            sx = fr.image.width / max(1, fr0.image.width)
            sy = fr.image.height / max(1, fr0.image.height)
            for n in nodes_needed:
                if n in fr.rig_nodes:
                    continue
                if n in fr0.rig_nodes:
                    x0, y0 = fr0.rig_nodes[n]
                    fr.rig_nodes[n] = (x0 * sx, y0 * sy)
        # if still missing, fallback center
        cx, cy = fr.image.width * 0.5, fr.image.height * 0.5
        for n in nodes_needed:
            if n not in fr.rig_nodes:
                fr.rig_nodes[n] = (cx, cy)
        fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, self.topology_bones)
        self.frames[idx] = fr

    def sync_all_frames_to_topology(self):
        if self.topology_bones is None:
            return
        for i in range(len(self.frames)):
            self._apply_topology_to_frame(i)

    def delete_topology_node(self, node_name: str):
        if self.topology_bones is None:
            self.initialize_topology_from_frame1()
        if self.topology_bones is None:
            return
        self.topology_bones = [(a, b) for (a, b) in self.topology_bones if a != node_name and b != node_name]
        for i in range(len(self.frames)):
            fr = self.frames[i]
            if fr.rig_nodes is not None and node_name in fr.rig_nodes:
                del fr.rig_nodes[node_name]
            if fr.rig_nodes is not None:
                fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, self.topology_bones)
            self.frames[i] = fr
        self.sync_all_frames_to_topology()
        self.mark_rig_dirty()
        self.refresh_ui()

    def split_topology_bone(self, a: str, b: str):
        if self.topology_bones is None:
            self.initialize_topology_from_frame1()
        if self.topology_bones is None:
            return
        # Find all matching edges regardless of direction.
        match_idxs = [
            i
            for i, (u, v) in enumerate(self.topology_bones)
            if (u == a and v == b) or (u == b and v == a)
        ]
        if not match_idxs:
            return

        # Preserve orientation from the first matching edge.
        first_u, first_v = self.topology_bones[match_idxs[0]]
        bone = (first_u, first_v)

        # Keep color channel coherent with the split segment.
        if bone[0].startswith("l_") or bone[1].startswith("l_"):
            prefix = "l_n"
        elif bone[0].startswith("r_") or bone[1].startswith("r_"):
            prefix = "r_n"
        else:
            prefix = "c_n"
        new_node = f"{prefix}{self.topology_new_node_counter}"
        self.topology_new_node_counter += 1

        # Remove every old A-B occurrence, then insert the split edges once.
        self.topology_bones = [
            (u, v)
            for i, (u, v) in enumerate(self.topology_bones)
            if i not in match_idxs
        ]
        insert_at = match_idxs[0]
        self.topology_bones[insert_at:insert_at] = [(bone[0], new_node), (new_node, bone[1])]

        for i in range(len(self.frames)):
            fr = self.frames[i]
            if fr.rig_nodes is None:
                continue
            if bone[0] in fr.rig_nodes and bone[1] in fr.rig_nodes:
                ax, ay = fr.rig_nodes[bone[0]]
                bx, by = fr.rig_nodes[bone[1]]
                fr.rig_nodes[new_node] = ((ax + bx) * 0.5, (ay + by) * 0.5)
            if fr.image is not None and new_node not in fr.rig_nodes:
                fr.rig_nodes[new_node] = (fr.image.width * 0.5, fr.image.height * 0.5)
            fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, self.topology_bones)
            self.frames[i] = fr
        self.sync_all_frames_to_topology()
        self.mark_rig_dirty()
        self.log(f"Segmento spezzato: {bone[0]}-{bone[1]} -> {bone[0]}-{new_node}-{bone[1]}")
        self.refresh_ui()

    def merge_topology_nodes(self, src_node: str, dst_node: str):
        if src_node == dst_node:
            return
        if self.topology_bones is None:
            self.initialize_topology_from_frame1()
        if self.topology_bones is None:
            return

        # Rewrite edges src -> dst.
        rewritten: List[Tuple[str, str]] = []
        for a, b in self.topology_bones:
            na = dst_node if a == src_node else a
            nb = dst_node if b == src_node else b
            if na == nb:
                continue
            rewritten.append((na, nb))

        # Remove duplicate undirected edges while preserving order.
        uniq: List[Tuple[str, str]] = []
        seen = set()
        for a, b in rewritten:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((a, b))
        self.topology_bones = uniq

        for i in range(len(self.frames)):
            fr = self.frames[i]
            if fr.rig_nodes is None:
                continue
            if src_node in fr.rig_nodes and dst_node in fr.rig_nodes:
                # Keep the dragged node position as final merged joint location.
                fr.rig_nodes[dst_node] = fr.rig_nodes[src_node]
            elif src_node in fr.rig_nodes and dst_node not in fr.rig_nodes:
                fr.rig_nodes[dst_node] = fr.rig_nodes[src_node]
            if src_node in fr.rig_nodes:
                del fr.rig_nodes[src_node]
            fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, self.topology_bones)
            self.frames[i] = fr

        self.sync_all_frames_to_topology()
        self.mark_rig_dirty()
        self.log(f"Nodi unificati: {src_node} -> {dst_node}")
        self.refresh_ui()

    def create_default_rig(self, img: Image.Image, fr: FrameData) -> Optional[Dict[str, Tuple[float, float]]]:
        profile = self.resolve_rig_profile(fr)
        if profile == "animal":
            return generate_animal_rig(img)
        if profile == "object":
            return generate_object_rig(img)
        return generate_bbox_rig(img)

    def create_rig_from_frame1_template(self, img: Image.Image) -> Optional[Dict[str, Tuple[float, float]]]:
        fr0 = self.frames[0]
        if fr0.image is None or fr0.rig_nodes is None:
            return None
        sx = img.width / max(1, fr0.image.width)
        sy = img.height / max(1, fr0.image.height)
        return {n: (x * sx, y * sy) for n, (x, y) in fr0.rig_nodes.items()}

    def set_frame_image(self, idx: int, img: Image.Image):
        cap = self.frames[idx].caption
        key_desc = self.frames[idx].key_description
        if self.use_gemini_caption.isChecked() and self.gemini_api_key.text().strip():
            try:
                cap = gemini_describe_image(
                    self.gemini_api_key.text().strip(),
                    self.gemini_model.text().strip() or "gemini-2.5-flash",
                    preprocess_for_caption(img),
                    hint=self.caption_hint.toPlainText().strip(),
                )
                self.log(f"[caption] Auto descrizione frame {idx + 1} con Gemini")
            except Exception as e:
                cap = cap or ""
                self.log(f"[caption] Errore Gemini su frame {idx + 1}: {e}")
        elif CAPTION_OK and (self.frames[idx].is_keyframe or self.frames[idx].rig_profile == "auto"):
            try:
                cap = caption_image(img)
                self.log(f"[caption] Auto descrizione frame {idx + 1} con fallback locale")
            except Exception:
                cap = ""
                self.log(f"[caption] Fallback locale fallito sul frame {idx + 1}")
        self.frames[idx].caption = cap
        self.frames[idx].detected_profile = infer_subject_profile(cap)
        if idx == 0:
            rig = self.create_default_rig(img, self.frames[idx])
        else:
            rig = self.create_rig_from_frame1_template(img)
            if rig is None:
                rig = self.create_default_rig(img, self.frames[idx])
        if self.frames[idx].is_keyframe and not key_desc:
            key_desc = cap
        self.frames[idx].image = img
        self.frames[idx].image_offset = (0, 0)
        self.frames[idx].generated = None
        self.frames[idx].rig_nodes = rig
        if idx == 0:
            self.initialize_topology_from_frame1()
        self._apply_topology_to_frame(idx)
        if idx == 0:
            self.sync_all_frames_to_topology()
        self.frames[idx].key_description = key_desc if self.frames[idx].is_keyframe else ""
        if rig is None:
            self.log(f"Frame {idx + 1}: rig non rilevato")
        self.mark_rig_dirty()
        self.refresh_ui()

    def refresh_preview(self):
        self.preview.update()

    def _motion_params_from_prompt(self) -> Tuple[float, float, float]:
        text = (self.prompt.toPlainText() or "").lower()
        amp = 0.04
        bob = 0.03
        swing = 0.08
        if "salta" in text or "jump" in text:
            bob = 0.08
            swing = 0.06
        if "corre" in text or "run" in text:
            swing = 0.16
            bob = 0.05
        if "cammina" in text or "walk" in text:
            swing = 0.11
            bob = 0.03
        if "idle" in text or "fermo" in text:
            amp = 0.01
            bob = 0.01
            swing = 0.02
        return amp, bob, swing

    def simulate_rig_animation(self):
        key_idxs = [i for i, fr in enumerate(self.frames) if fr.is_keyframe and fr.rig_nodes is not None]
        if not key_idxs:
            key_idxs = [i for i, fr in enumerate(self.frames) if fr.rig_nodes is not None]
        if not key_idxs:
            QMessageBox.warning(self, "Attenzione", "Carica almeno un frame chiave")
            return

        amp, bob, swing = self._motion_params_from_prompt()
        for i in range(len(self.frames)):
            fr = self.frames[i]
            if fr.is_keyframe:
                if fr.rig_nodes is not None:
                    bones = self.get_frame_bones(fr)
                    fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, bones)
                continue
            prev = max([k for k in key_idxs if k < i], default=None)
            nxt = min([k for k in key_idxs if k > i], default=None)
            if prev is not None and nxt is not None:
                t = (i - prev) / (nxt - prev)
                nodes = {}
                common = set(self.frames[prev].rig_nodes.keys()) & set(self.frames[nxt].rig_nodes.keys())
                for n in common:
                    px, py = self.frames[prev].rig_nodes[n]
                    nx, ny = self.frames[nxt].rig_nodes[n]
                    nodes[n] = (px + (nx - px) * t, py + (ny - py) * t)
                if not nodes:
                    nodes = dict(self.frames[prev].rig_nodes)
                # Add mild procedural motion from prompt for non-key segments.
                src_img = self.frames[prev].image or self.frames[nxt].image
                if src_img is not None:
                    ph = (i / max(1, len(self.frames) - 1)) * 2.0 * math.pi
                    dx = math.sin(ph) * src_img.width * amp
                    dy = math.sin(ph + math.pi / 2) * src_img.height * bob
                    for n in list(nodes.keys()):
                        x, y = nodes[n]
                        if n.endswith("hand") or n.endswith("foot"):
                            nodes[n] = (x + dx * swing, y + dy)
                fr.rig_nodes = nodes
                fr.detected_profile = self.frames[prev].detected_profile
            else:
                src = self.frames[prev if prev is not None else nxt]
                nodes = dict(src.rig_nodes)
                # Single-keyframe case: synthesize motion over timeline.
                src_img = src.image or self.frames[0].image
                if src_img is not None:
                    ph = (i / max(1, len(self.frames) - 1)) * 2.0 * math.pi
                    dx = math.sin(ph) * src_img.width * amp
                    dy = math.sin(ph + math.pi / 2) * src_img.height * bob
                    for n in list(nodes.keys()):
                        x, y = nodes[n]
                        if n.endswith("hand") or n.endswith("foot"):
                            nodes[n] = (x + dx * (1.0 + swing), y + dy)
                        elif n in ("hip", "spine", "neck", "center"):
                            nodes[n] = (x + dx * 0.35, y + dy * 0.35)
                fr.rig_nodes = nodes
                fr.detected_profile = src.detected_profile
            bones = self.get_frame_bones(fr)
            fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, bones)
        self.mark_rig_dirty()
        self.refresh_ui()
        self.log("Simulazione rig completata (keyframe + prompt)")

    def fill_missing_poses(self):
        self.simulate_rig_animation()

    def generate_rig_with_ai(self):
        if not self.gemini_api_key.text().strip():
            QMessageBox.warning(self, "AI rig", "Inserisci la API key Gemini per usare AI rig.")
            self.log("[AI rig] Errore: API key mancante")
            return
        fr0 = self.frames[0]
        if fr0.rig_nodes is None or fr0.image is None:
            QMessageBox.warning(self, "AI rig", "Carica e prepara il frame 1 prima di generare l'animazione.")
            self.log("[AI rig] Errore: frame 1 non pronto (rig o immagine mancante)")
            return
        bones = self.get_frame_bones(fr0)
        key_descs = {i: fr.key_description for i, fr in enumerate(self.frames) if fr.is_keyframe and fr.key_description}
        key_nodes = {}
        for i, fr in enumerate(self.frames):
            if not fr.is_keyframe or fr.rig_nodes is None:
                continue
            if fr.image is None:
                continue
            w_i, h_i = fr.image.size
            norm = {}
            for n, (x, y) in fr.rig_nodes.items():
                nx = 0.0 if w_i <= 1 else max(0.0, min(1.0, x / (w_i - 1)))
                ny = 0.0 if h_i <= 1 else max(0.0, min(1.0, y / (h_i - 1)))
                norm[n] = (nx, ny)
            key_nodes[i] = norm
        self.log("[AI rig] Richiesta inviata a Gemini")
        try:
            frames_norm = gemini_generate_rig_animation(
                self.gemini_api_key.text().strip(),
                self.gemini_model.text().strip() or "gemini-2.5-flash",
                fr0.rig_nodes,
                bones,
                len(self.frames),
                self.prompt.toPlainText().strip(),
                key_descs,
                key_nodes,
            )
        except Exception as e:
            self.log(f"[AI rig] Errore: {e}")
            QMessageBox.warning(self, "AI rig", f"Errore: {e}")
            return

        w, h = fr0.image.size
        for i, frame_nodes in enumerate(frames_norm):
            if not frame_nodes:
                continue
            abs_nodes = {}
            for n, (nx, ny) in frame_nodes.items():
                x = max(0.0, min(1.0, nx)) * (w - 1)
                y = max(0.0, min(1.0, ny)) * (h - 1)
                abs_nodes[n] = (x, y)
            if self.frames[i].is_keyframe and self.frames[i].rig_nodes is not None:
                continue
            self.frames[i].rig_nodes = abs_nodes
            self.frames[i].bone_lengths = compute_bone_lengths(abs_nodes, self.get_frame_bones(self.frames[i]))
        self.mark_rig_dirty()
        self.refresh_ui()
        self.log("[AI rig] Animazione rig generata")

    def generate_final_frames(self):
        if not self.rig_approved:
            QMessageBox.warning(self, "Attenzione", "Approva prima il rig (pulsante 'approva rig').")
            return
        idx0 = next((i for i, fr in enumerate(self.frames) if fr.image is not None), None)
        if idx0 is None:
            QMessageBox.warning(self, "Attenzione", "Carica almeno un frame con sprite")
            return

        for i, fr in enumerate(self.frames):
            if fr.rig_nodes is None:
                QMessageBox.warning(self, "Attenzione", f"Manca il rig nel frame {i + 1}")
                return

        use_cuda = torch.cuda.is_available()
        try:
            pipe = load_pipe("runwayml/stable-diffusion-v1-5", "lllyasviel/control_v11p_sd15_openpose", use_cuda)
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Caricamento modelli fallito: {e}")
            return

        base_img = self.frames[idx0].image.convert("RGB")
        plan = calc_gen_size(base_img.width, base_img.height, "resize_down")

        generator = torch.Generator(device="cuda" if use_cuda else "cpu").manual_seed(42)

        for i, fr in enumerate(self.frames):
            try:
                bones = self.get_frame_bones(fr)
                pose_map = rig_to_pose_map(fr.rig_nodes, fr.image.size if fr.image is not None else base_img.size, bones)
                pose_in = resize_or_pad(pose_map.convert("RGB"), plan)
                out = pipe(
                    prompt=self.prompt.toPlainText().strip() or "sprite",
                    negative_prompt=self.negative.toPlainText().strip(),
                    image=pose_in,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                    width=plan.gen_w,
                    height=plan.gen_h,
                ).images[0]
                out = crop_or_resize_back(out, plan)
                fr.generated = out.convert("RGBA")
                self.log(f"Generato frame {i + 1}/{len(self.frames)}")
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Generazione frame {i + 1} fallita: {e}")
                return

        self.refresh_ui()

        save_path, _ = QFileDialog.getSaveFileName(self, "Salva zip", str(Path.cwd() / "frames.zip"), "Zip (*.zip)")
        if save_path:
            self.save_generated_zip(save_path)
            self.log(f"Zip salvato: {save_path}")

    def save_generated_zip(self, path: str):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, fr in enumerate(self.frames):
                img = fr.generated if fr.generated is not None else (compose_with_offset(fr.image, fr.image_offset) if fr.image is not None else None)
                if img is None:
                    continue
                b = io.BytesIO()
                img.save(b, format="PNG")
                zf.writestr(f"frame_{i+1:03d}.png", b.getvalue())


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()




