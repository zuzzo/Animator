import io
import copy
import math
import sys
import zipfile
import base64
import json
import warnings
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Silence known third-party startup warnings that do not affect app behavior.
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit_.* in registry.*",
    category=UserWarning,
)

import numpy as np
import requests
import torch
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter
from PySide6.QtCore import QMimeData, QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QDrag, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
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
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
)

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
    image_unkeyed: Optional[Image.Image] = None
    image_offset: Tuple[int, int] = (0, 0)
    rig_nodes: Optional[Dict[str, Tuple[float, float]]] = None
    bone_lengths: Optional[Dict[Tuple[str, str], float]] = None
    is_keyframe: bool = False
    key_description: str = ""
    caption: str = ""
    rig_profile: str = "auto"  # auto | human | animal | object
    detected_profile: str = "human"
    generated: Optional[Image.Image] = None
    bone_correspondence: Optional[Dict[Tuple[str, str], str]] = None


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

SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_CONTROLNET_OPENPOSE_ID = "xinsir/controlnet-openpose-sdxl-1.0"


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


def build_subject_mask(img: Optional[Image.Image]) -> Optional[np.ndarray]:
    if img is None:
        return None
    rgba = np.array(img.convert("RGBA"))
    alpha = rgba[..., 3]
    mask = alpha > 10
    if mask.any():
        return mask
    rgb = rgba[..., :3].astype(np.int16)
    bg = np.median(rgb.reshape(-1, 3), axis=0)
    dist = np.abs(rgb - bg).sum(axis=2)
    return dist > 24


def segment_coverage(mask: np.ndarray, a: Tuple[float, float], b: Tuple[float, float], samples: int = 28) -> float:
    h, w = mask.shape
    hits = 0
    total = max(2, samples)
    for s in range(total):
        t = s / float(total - 1)
        x = int(round(a[0] + (b[0] - a[0]) * t))
        y = int(round(a[1] + (b[1] - a[1]) * t))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        if mask[y, x]:
            hits += 1
    return hits / float(total)


def estimate_bone_raster_reference(
    nodes: Dict[str, Tuple[float, float]],
    bones: List[Tuple[str, str]],
    img: Optional[Image.Image],
) -> Dict[Tuple[str, str], float]:
    mask = build_subject_mask(img)
    out: Dict[Tuple[str, str], float] = {}
    for a, b in bones:
        if a not in nodes or b not in nodes:
            continue
        if mask is None:
            out[(a, b)] = 0.45
        else:
            out[(a, b)] = float(segment_coverage(mask, nodes[a], nodes[b]))
    return out


def derive_node_mobility(
    nodes: Dict[str, Tuple[float, float]],
    bones: List[Tuple[str, str]],
    bone_ref: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    acc: Dict[str, float] = {n: 0.0 for n in nodes.keys()}
    cnt: Dict[str, int] = {n: 0 for n in nodes.keys()}
    for a, b in bones:
        cov = bone_ref.get((a, b), 0.45)
        if a in acc:
            acc[a] += cov
            cnt[a] += 1
        if b in acc:
            acc[b] += cov
            cnt[b] += 1
    out: Dict[str, float] = {}
    for n in nodes.keys():
        avg_cov = acc[n] / max(1, cnt[n])
        out[n] = max(0.35, min(1.0, 1.05 - avg_cov * 0.70))
    return out


def _bone_feature(
    nodes: Dict[str, Tuple[float, float]],
    bone: Tuple[str, str],
    canvas_size: Tuple[int, int],
    mask: Optional[np.ndarray],
) -> Optional[Tuple[float, float, float, float, float]]:
    a, b = bone
    if a not in nodes or b not in nodes:
        return None
    w, h = max(1, canvas_size[0]), max(1, canvas_size[1])
    ax, ay = nodes[a]
    bx, by = nodes[b]
    mx = ((ax + bx) * 0.5) / w
    my = ((ay + by) * 0.5) / h
    ln = math.hypot((bx - ax) / w, (by - ay) / h)
    ang = (math.atan2(by - ay, bx - ax) + math.pi) / (2.0 * math.pi)
    cov = 0.5
    if mask is not None:
        cov = float(segment_coverage(mask, (ax, ay), (bx, by)))
    return mx, my, ln, ang, cov


def _angle_wrap_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def infer_bone_correspondence(
    nodes: Dict[str, Tuple[float, float]],
    bones: List[Tuple[str, str]],
    template_nodes: Dict[str, Tuple[float, float]],
    template_bones: List[Tuple[str, str]],
    canvas_size: Tuple[int, int],
    mask: Optional[np.ndarray],
) -> Dict[Tuple[str, str], str]:
    def _norm_bone(b):
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return (str(b[0]), str(b[1]))
        return None

    out: Dict[Tuple[str, str], str] = {}
    cur_feat: Dict[Tuple[str, str], Tuple[float, float, float, float, float]] = {}
    tpl_feat: Dict[Tuple[str, str], Tuple[float, float, float, float, float]] = {}
    norm_bones = [nb for nb in (_norm_bone(b) for b in bones) if nb is not None]
    norm_tpl_bones = [nb for nb in (_norm_bone(b) for b in template_bones) if nb is not None]
    for b in norm_bones:
        f = _bone_feature(nodes, b, canvas_size, mask)
        if f is not None:
            cur_feat[b] = f
    for b in norm_tpl_bones:
        f = _bone_feature(template_nodes, b, canvas_size, mask)
        if f is not None:
            tpl_feat[b] = f
    if not cur_feat or not tpl_feat:
        return out

    candidates: List[Tuple[float, Tuple[str, str], Tuple[str, str]]] = []
    for cb, cf in cur_feat.items():
        for tb, tf in tpl_feat.items():
            side_pen = 0.0
            c_left = cb[0].startswith("l_") or cb[1].startswith("l_")
            c_right = cb[0].startswith("r_") or cb[1].startswith("r_")
            t_left = tb[0].startswith("l_") or tb[1].startswith("l_")
            t_right = tb[0].startswith("r_") or tb[1].startswith("r_")
            if (c_left and t_right) or (c_right and t_left):
                side_pen = 0.45
            md = math.hypot(cf[0] - tf[0], cf[1] - tf[1])
            ld = abs(cf[2] - tf[2])
            ad = _angle_wrap_dist(cf[3], tf[3])
            cd = abs(cf[4] - tf[4])
            score = md + 0.35 * ld + 0.30 * ad + 0.20 * cd + side_pen
            candidates.append((score, cb, tb))

    candidates.sort(key=lambda x: x[0])
    used_cur = set()
    used_tpl = set()
    for _, cb, tb in candidates:
        if cb in used_cur or tb in used_tpl:
            continue
        used_cur.add(cb)
        used_tpl.add(tb)
        out[cb] = f"{tb[0]}->{tb[1]}"

    for b in norm_bones:
        if b not in out:
            out[b] = f"{b[0]}->{b[1]}"
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


def apply_chroma_key(img: Image.Image, key_rgb: tuple, tol: int, edge_connected: bool = False) -> Image.Image:
    rgba = img.convert("RGBA")
    data = np.array(rgba)
    r = data[..., 0].astype(np.int16)
    g = data[..., 1].astype(np.int16)
    b = data[..., 2].astype(np.int16)
    kr, kg, kb = key_rgb
    mask = (
        (np.abs(r - kr) <= tol)
        & (np.abs(g - kg) <= tol)
        & (np.abs(b - kb) <= tol)
    )
    alpha = data[..., 3]
    if mask.shape != alpha.shape and mask.T.shape == alpha.shape:
        mask = mask.T
    if edge_connected:
        h, w = mask.shape
        bg = np.zeros_like(mask, dtype=bool)
        q = deque()
        for x in range(w):
            if mask[0, x]:
                bg[0, x] = True
                q.append((0, x))
            if mask[h - 1, x] and not bg[h - 1, x]:
                bg[h - 1, x] = True
                q.append((h - 1, x))
        for y in range(h):
            if mask[y, 0] and not bg[y, 0]:
                bg[y, 0] = True
                q.append((y, 0))
            if mask[y, w - 1] and not bg[y, w - 1]:
                bg[y, w - 1] = True
                q.append((y, w - 1))
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not mask[ny, nx] or bg[ny, nx]:
                    continue
                bg[ny, nx] = True
                q.append((ny, nx))
        mask = bg
    data[..., 3] = np.where(mask, 0, alpha)
    return Image.fromarray(data)


def pick_effective_key_color(img: Image.Image, preferred_rgb: Tuple[int, int, int], tol: int) -> Tuple[int, int, int]:
    rgb = np.array(img.convert("RGB"))
    top = rgb[0, :, :]
    bottom = rgb[-1, :, :]
    left = rgb[:, 0, :]
    right = rgb[:, -1, :]
    border = np.concatenate([top, bottom, left, right], axis=0).astype(np.int16)
    p = np.array(preferred_rgb, dtype=np.int16)
    dist = np.abs(border - p).sum(axis=1)
    hits = float(np.mean(dist <= max(3, tol * 3)))
    if hits >= 0.01:
        return preferred_rgb
    med = np.median(border, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


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


def quantize_to_reference_palette(
    img: Image.Image, ref_img: Image.Image, max_colors: int = 64
) -> Image.Image:
    src_rgb = img.convert("RGB")
    ref_rgb = ref_img.convert("RGB")
    palette_src = ref_rgb.quantize(colors=max(2, min(256, int(max_colors))), method=Image.MEDIANCUT)
    quant = src_rgb.quantize(palette=palette_src, dither=Image.NONE)
    return quant.convert("RGBA")


def _rig_anchor(nodes: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    for n in ("hip", "center", "neck", "spine", "pivot"):
        if n in nodes:
            return nodes[n]
    if not nodes:
        return None
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def estimate_global_rig_shift(
    prev_nodes: Dict[str, Tuple[float, float]], curr_nodes: Dict[str, Tuple[float, float]]
) -> Tuple[int, int]:
    pa = _rig_anchor(prev_nodes)
    ca = _rig_anchor(curr_nodes)
    if pa is None or ca is None:
        return 0, 0
    return int(round(ca[0] - pa[0])), int(round(ca[1] - pa[1]))


def rig_motion_score(
    prev_nodes: Dict[str, Tuple[float, float]],
    curr_nodes: Dict[str, Tuple[float, float]],
    canvas_size: Tuple[int, int],
) -> float:
    common = set(prev_nodes.keys()) & set(curr_nodes.keys())
    if not common:
        return 0.0
    w, h = max(1, int(canvas_size[0])), max(1, int(canvas_size[1]))
    diag = max(1.0, math.hypot(w, h))
    acc = 0.0
    for n in common:
        px, py = prev_nodes[n]
        cx, cy = curr_nodes[n]
        acc += math.hypot(cx - px, cy - py)
    mean_disp = acc / max(1, len(common))
    # Normalize so typical limb motion yields a noticeable boost.
    score = mean_disp / (0.14 * diag)
    return max(0.0, min(1.0, score))


def warp_sprite_with_rig(
    base_img: Image.Image,
    base_nodes: Dict[str, Tuple[float, float]],
    target_nodes: Dict[str, Tuple[float, float]],
) -> Image.Image:
    src = np.array(base_img.convert("RGBA"))
    h, w = src.shape[:2]
    common = [n for n in base_nodes.keys() if n in target_nodes]
    if not common:
        return base_img.convert("RGBA")

    anchors = np.array([[base_nodes[n][0], base_nodes[n][1]] for n in common], dtype=np.float32)
    deltas = np.array(
        [[target_nodes[n][0] - base_nodes[n][0], target_nodes[n][1] - base_nodes[n][1]] for n in common],
        dtype=np.float32,
    )
    deltas[:, 0] = np.clip(deltas[:, 0], -0.40 * w, 0.40 * w)
    deltas[:, 1] = np.clip(deltas[:, 1], -0.40 * h, 0.40 * h)
    alpha = src[..., 3] > 0
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros_like(src)

    # Move connected alpha components as rigid 2D cutouts to avoid stretching artifacts.
    for y0 in range(h):
        for x0 in range(w):
            if not alpha[y0, x0] or visited[y0, x0]:
                continue
            q = deque([(y0, x0)])
            visited[y0, x0] = True
            pts: List[Tuple[int, int]] = []
            while q:
                y, x = q.popleft()
                pts.append((y, x))
                if y > 0 and alpha[y - 1, x] and not visited[y - 1, x]:
                    visited[y - 1, x] = True
                    q.append((y - 1, x))
                if y + 1 < h and alpha[y + 1, x] and not visited[y + 1, x]:
                    visited[y + 1, x] = True
                    q.append((y + 1, x))
                if x > 0 and alpha[y, x - 1] and not visited[y, x - 1]:
                    visited[y, x - 1] = True
                    q.append((y, x - 1))
                if x + 1 < w and alpha[y, x + 1] and not visited[y, x + 1]:
                    visited[y, x + 1] = True
                    q.append((y, x + 1))

            xs = np.array([p[1] for p in pts], dtype=np.float32)
            ys = np.array([p[0] for p in pts], dtype=np.float32)
            cx = float(xs.mean())
            cy = float(ys.mean())
            d2 = (anchors[:, 0] - cx) ** 2 + (anchors[:, 1] - cy) ** 2
            idx = int(np.argmin(d2))
            dx = int(round(float(deltas[idx, 0])))
            dy = int(round(float(deltas[idx, 1])))

            for y, x in pts:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if src[y, x, 3] >= out[ny, nx, 3]:
                    out[ny, nx] = src[y, x]

    return Image.fromarray(out, mode="RGBA")


@lru_cache(maxsize=1)
def load_openpose() -> OpenposeDetector:
    return OpenposeDetector.from_pretrained("lllyasviel/Annotators")


@lru_cache(maxsize=4)
def load_pipe(base_model_id: str, controlnet_id: str, use_cuda: bool):
    dtype = torch.float16 if use_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    base_path = Path(base_model_id)
    if base_path.is_file() and base_path.suffix.lower() in (".safetensors", ".ckpt"):
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            str(base_path),
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base_model_id, controlnet=controlnet, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers is optional; keep running with standard attention.
            pass
        pipe.to("cuda")
    else:
        pipe.enable_attention_slicing()
        pipe.to("cpu")
    return pipe


@lru_cache(maxsize=4)
def load_pipe_img2img(base_model_id: str, controlnet_id: str, use_cuda: bool):
    dtype = torch.float16 if use_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    base_path = Path(base_model_id)
    if base_path.is_file() and base_path.suffix.lower() in (".safetensors", ".ckpt"):
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            str(base_path),
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    else:
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(base_model_id, controlnet=controlnet, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers is optional; keep running with standard attention.
            pass
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


def normalize_gemini_api_key(api_key: str) -> str:
    key = (api_key or "").strip().strip("\"").strip("'")
    # Remove hidden whitespace/newline characters accidentally copied with the key.
    key = "".join(ch for ch in key if ch.isprintable() and not ch.isspace())
    return key


def format_gemini_http_error(res: requests.Response) -> str:
    try:
        payload = res.json()
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        status = err.get("status", "")
        message = (err.get("message", "") or "").strip()
        reasons = []
        details = err.get("details", [])
        if isinstance(details, list):
            for d in details:
                if isinstance(d, dict):
                    r = d.get("reason")
                    if isinstance(r, str) and r and r not in reasons:
                        reasons.append(r)
        reason_txt = ",".join(reasons)
        parts = [f"Gemini HTTP {res.status_code}"]
        if status:
            parts.append(f"status={status}")
        if reason_txt:
            parts.append(f"reason={reason_txt}")
        if message:
            parts.append(f"message={message}")
        return " ".join(parts)
    except Exception:
        body = (res.text or "").strip().replace("\n", " ")
        body = body[:500]
        return f"Gemini HTTP {res.status_code} body={body}"


def gemini_describe_image(api_key: str, model: str, img: Image.Image, hint: str = "") -> str:
    api_key = normalize_gemini_api_key(api_key)
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
        params={"key": api_key},
        json=body,
        timeout=60,
    )
    if not res.ok:
        raise RuntimeError(format_gemini_http_error(res))
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
    api_key = normalize_gemini_api_key(api_key)
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
                params={"key": api_key},
                json=body,
                timeout=120,
            )
            if not res.ok:
                raise RuntimeError(format_gemini_http_error(res))
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
            if r > 0:
                onion = Image.new("RGBA", base.size, (0, 0, 0, 0))
                cur = self.parent_win.selected_index
                lo = max(0, cur - r)
                hi = min(len(self.parent_win.frames), cur + r + 1)
                for k in range(lo, hi):
                    if k == cur:
                        continue
                    d = abs(k - cur)
                    other = self.parent_win.frames[k].generated or self.parent_win.frames[k].image
                    if other is None:
                        continue
                    if self.parent_win.frames[k].generated is None:
                        other = compose_with_offset(other, self.parent_win.frames[k].image_offset)
                    ov = other.convert("RGBA")
                    if ov.size != base.size:
                        ov = ov.resize(base.size, Image.NEAREST)
                    max_alpha = 140
                    min_alpha = 25
                    fall = (r - d + 1) / max(1, r)
                    alpha = int(min_alpha + (max_alpha - min_alpha) * max(0.0, min(1.0, fall)))
                    ov.putalpha(alpha)
                    onion = Image.alpha_composite(onion, ov)
                base = Image.alpha_composite(onion, base)
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
            # Onion skin for rig (semi-transparent previous/next frames).
            if self.parent_win.chk_onion.isChecked():
                r = self.parent_win.onion_range.value()
                if r > 0:
                    cur = self.parent_win.selected_index
                    lo = max(0, cur - r)
                    hi = min(len(self.parent_win.frames), cur + r + 1)
                    for k in range(lo, hi):
                        if k == cur:
                            continue
                        other = self.parent_win.frames[k]
                        if other.rig_nodes is None:
                            continue
                        d = abs(k - cur)
                        max_alpha = 140
                        min_alpha = 25
                        fall = (r - d + 1) / max(1, r)
                        alpha = int(min_alpha + (max_alpha - min_alpha) * max(0.0, min(1.0, fall)))
                        bones = self._frame_bones(other)
                        used_nodes = set()
                        for a, b in bones:
                            if a not in other.rig_nodes or b not in other.rig_nodes:
                                continue
                            used_nodes.add(a)
                            used_nodes.add(b)
                            ax, ay = other.rig_nodes[a]
                            bx, by = other.rig_nodes[b]
                            awx, awy = self._image_to_widget(ax, ay, rect, img.size)
                            bwx, bwy = self._image_to_widget(bx, by, rect, img.size)
                            if a.startswith("l_") or b.startswith("l_"):
                                c = QColor(*RIG_COLORS["left"])
                            elif a.startswith("r_") or b.startswith("r_"):
                                c = QColor(*RIG_COLORS["right"])
                            else:
                                c = QColor(*RIG_COLORS["center"])
                            c.setAlpha(alpha)
                            painter.setPen(QPen(c, 2))
                            painter.drawLine(awx, awy, bwx, bwy)
                        for n, (x, y) in other.rig_nodes.items():
                            if n not in used_nodes:
                                continue
                            wx, wy = self._image_to_widget(x, y, rect, img.size)
                            if n.startswith("l_"):
                                c = QColor(*RIG_COLORS["left"])
                            elif n.startswith("r_"):
                                c = QColor(*RIG_COLORS["right"])
                            else:
                                c = QColor(*RIG_COLORS["center"])
                            c.setAlpha(alpha)
                            painter.setPen(QPen(QColor(255, 255, 255, alpha), 0))
                            painter.setBrush(QBrush(QColor(255, 255, 255, alpha)))
                            painter.drawEllipse(wx - 6, wy - 6, 12, 12)
                            painter.setPen(QPen(QColor(15, 23, 42, alpha), 1))
                            painter.setBrush(QBrush(c))
                            painter.drawEllipse(wx - 4, wy - 4, 8, 8)

            bones = self._frame_bones(fr)
            corr_map: Dict[Tuple[str, str], str] = {}
            show_map = hasattr(self.parent_win, "chk_bone_map") and self.parent_win.chk_bone_map.isChecked()
            if show_map:
                corr_map = fr.bone_correspondence or self.parent_win.analyze_frame_bone_correspondence(
                    self.parent_win.selected_index, log_result=False
                )
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
                if show_map:
                    label = corr_map.get((a, b), f"{a}->{b}")
                    mx = int((awx + bwx) * 0.5)
                    my = int((awy + bwy) * 0.5)
                    text_rect = QRect(mx - 58, my - 18, 116, 14)
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                    painter.drawRect(text_rect)
                    painter.setPen(QPen(QColor("#0f172a"), 1))
                    painter.drawText(text_rect.adjusted(2, 0, -2, 0), Qt.AlignCenter, label)

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
        self.playing: bool = False
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(120)
        self.play_timer.timeout.connect(self.on_play_tick)

        self.detail_dialog = DetailDialog(self)
        self.build_menu()

        self.build_ui()
        self.load_config()
        self.refresh_ui()
        self.log_runtime_device()

    def build_menu(self):
        menu = self.menuBar().addMenu("Progetto")
        menu.addAction("Salva progetto", self.save_project)
        menu.addAction("Importa progetto", self.load_project)
        menu.addSeparator()
        menu.addAction("Salva config", self.save_config)

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
        self.btn_generate_frames = QPushButton("genera raster")
        self.btn_generate_frames.clicked.connect(self.generate_final_frames)
        self.btn_generate_frames.setEnabled(True)
        self.gen_strength = QDoubleSpinBox()
        self.gen_strength.setRange(0.1, 0.95)
        self.gen_strength.setSingleStep(0.05)
        self.gen_strength.setValue(0.42)
        self.gen_strength.setPrefix("img2img strength ")
        self.control_weight = QDoubleSpinBox()
        self.control_weight.setRange(0.2, 2.5)
        self.control_weight.setSingleStep(0.1)
        self.control_weight.setValue(1.2)
        self.control_weight.setPrefix("control weight ")
        self.force_rig_warp = QCheckBox("forza rig-warp")
        self.force_rig_warp.setChecked(True)
        self.sdxl_base_model_path = QLineEdit("")
        self.sdxl_base_model_path.setPlaceholderText("Checkpoint SDXL locale (.safetensors/.ckpt) opzionale")
        self.btn_pick_sdxl_model = QPushButton("modello SDXL...")
        self.btn_pick_sdxl_model.clicked.connect(self.pick_sdxl_model_path)

        left_layout.addWidget(self.btn_generate_pose)
        left_layout.addWidget(self.btn_ai_rig)
        left_layout.addWidget(self.btn_generate_frames)
        left_layout.addWidget(self.gen_strength)
        left_layout.addWidget(self.control_weight)
        left_layout.addWidget(self.force_rig_warp)
        left_layout.addWidget(self.sdxl_base_model_path)
        left_layout.addWidget(self.btn_pick_sdxl_model)

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
        self.btn_play = QPushButton("play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self.toggle_play)
        right_top.addWidget(self.btn_play)
        right_top.addStretch(1)
        self.chk_sprite = QCheckBox("raster")
        self.chk_sprite.setChecked(True)
        self.chk_sprite.toggled.connect(self.refresh_preview)
        self.chk_pose = QCheckBox("scheletro")
        self.chk_pose.setChecked(True)
        self.chk_pose.toggled.connect(self.refresh_preview)
        self.chk_bone_map = QCheckBox("map ossa")
        self.chk_bone_map.setChecked(False)
        self.chk_bone_map.toggled.connect(self.refresh_preview)
        right_top.addWidget(self.chk_sprite)
        right_top.addWidget(self.chk_pose)
        right_top.addWidget(self.chk_bone_map)
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
        self.chk_onion.toggled.connect(self.refresh_preview)
        self.onion_range = QSpinBox()
        self.onion_range.valueChanged.connect(self.refresh_preview)
        self.onion_range.setRange(1, 10)
        self.onion_range.setValue(2)
        self.onion_range.setToolTip("Numero di frame prima/dopo da mostrare")
        self.preview_move_step = QSpinBox()
        self.preview_move_step.setRange(1, 64)
        self.preview_move_step.setValue(2)
        self.preview_move_step.setToolTip("Passo spostamento frecce (px)")
        self.eyedropper_tol = QSpinBox()
        self.eyedropper_tol.setRange(0, 80)
        self.eyedropper_tol.setValue(10)
        self.eyedropper_tol.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.eyedropper_tol.setFixedWidth(48)
        self.eyedropper_color_chip = QLabel()
        self.eyedropper_color_chip.setFixedSize(22, 22)
        self.eyedropper_color_chip.setToolTip("Ultimo colore selezionato")
        self._update_eyedropper_chip()
        def _spin_group(label_text: str, spin: QSpinBox) -> QWidget:
            wrap = QWidget()
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(label_text)
            btn_minus = QPushButton("-")
            btn_plus = QPushButton("+")
            btn_minus.setObjectName("spinbtn")
            btn_plus.setObjectName("spinbtn")
            btn_minus.setFixedSize(22, 22)
            btn_plus.setFixedSize(22, 22)
            btn_minus.setAutoRepeat(True)
            btn_plus.setAutoRepeat(True)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setFixedWidth(48)
            spin.setReadOnly(False)
            btn_minus.clicked.connect(
                lambda _=None, s=spin: s.setValue(max(s.minimum(), s.value() - s.singleStep()))
            )
            btn_plus.clicked.connect(
                lambda _=None, s=spin: s.setValue(min(s.maximum(), s.value() + s.singleStep()))
            )
            row.addWidget(lab)
            row.addWidget(btn_minus)
            row.addWidget(spin)
            row.addWidget(btn_plus)
            return wrap

        right_bottom.addWidget(self.chk_onion)
        right_bottom.addWidget(_spin_group("range", self.onion_range))
        right_bottom.addWidget(_spin_group("spostamento", self.preview_move_step))
        right_bottom.addWidget(_spin_group("tol", self.eyedropper_tol))
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
            "QPushButton#spinbtn { padding: 0px; min-width: 22px; min-height: 22px; }"
            "QCheckBox { color: #0f172a; spacing: 6px; }"
            "QSpinBox, QDoubleSpinBox, QComboBox { background: #ffffff; color: #0f172a; border: 1px solid #94a3b8; }"
            "QLabel#title { font-size: 48px; font-weight: 700; color: #334155; }"
            "QWidget#preview { background: #ffffff; border: 2px solid #94a3b8; }"
            "QSlider::groove:horizontal { height: 6px; background: #cbd5e1; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -5px 0; border-radius: 7px; }"
        )

    def log(self, msg: str):
        self.log_box.append(msg)

    def log_runtime_device(self):
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.log(f"[runtime] CUDA ON - GPU: {gpu_name}")
            else:
                self.log("[runtime] CUDA OFF - uso CPU")
        except Exception as e:
            self.log(f"[runtime] Stato CUDA non disponibile: {e}")

    def _config_path(self) -> Path:
        return Path.home() / ".animator" / "config.json"

    def load_config(self):
        try:
            cfg_path = self._config_path()
            if not cfg_path.exists():
                return
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            api_key = (data.get("gemini_api_key") or "").strip()
            if api_key:
                self.gemini_api_key.setText(api_key)
            model = (data.get("gemini_model") or "").strip()
            if model:
                self.gemini_model.setText(model)
            strength = data.get("gen_strength")
            if strength is not None:
                self.gen_strength.setValue(max(0.1, min(0.95, float(strength))))
            ctrl = data.get("control_weight")
            if ctrl is not None:
                self.control_weight.setValue(max(0.2, min(2.5, float(ctrl))))
            frw = data.get("force_rig_warp")
            if frw is not None:
                self.force_rig_warp.setChecked(bool(frw))
            mdl = (data.get("sdxl_base_model_path") or "").strip()
            if mdl:
                self.sdxl_base_model_path.setText(mdl)
        except Exception as e:
            self.log(f"Errore lettura config: {e}")

    def save_config(self):
        try:
            cfg_path = self._config_path()
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "gemini_api_key": self.gemini_api_key.text().strip(),
                "gemini_model": self.gemini_model.text().strip(),
                "gen_strength": float(self.gen_strength.value()),
                "control_weight": float(self.control_weight.value()),
                "force_rig_warp": bool(self.force_rig_warp.isChecked()),
                "sdxl_base_model_path": self.sdxl_base_model_path.text().strip(),
            }
            cfg_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as e:
            self.log(f"Errore salvataggio config: {e}")

    def _frame_to_project_dict(self, fr: FrameData, idx: int) -> Dict:
        return {
            "index": idx,
            "image": f"frames/{idx}_image.png" if fr.image is not None else None,
            "generated": f"frames/{idx}_generated.png" if fr.generated is not None else None,
            "image_offset": list(fr.image_offset),
            "rig_nodes": {k: [float(x), float(y)] for k, (x, y) in (fr.rig_nodes or {}).items()},
            "is_keyframe": bool(fr.is_keyframe),
            "key_description": fr.key_description,
            "caption": fr.caption,
            "rig_profile": fr.rig_profile,
            "detected_profile": fr.detected_profile,
        }

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva progetto",
            str(Path.cwd() / "progetto.animproj"),
            "Animator Project (*.animproj)",
        )
        if not path:
            return
        self.save_config()
        if not path.lower().endswith(".animproj"):
            path += ".animproj"

        project = {
            "version": 1,
            "prompt": self.prompt.toPlainText(),
            "negative": self.negative.toPlainText(),
            "caption_hint": self.caption_hint.toPlainText(),
            "use_gemini_caption": bool(self.use_gemini_caption.isChecked()),
            "gemini_model": self.gemini_model.text().strip(),
            "frame_count": len(self.frames),
            "selected_index": self.selected_index,
            "onion_enabled": bool(self.chk_onion.isChecked()),
            "onion_range": int(self.onion_range.value()),
            "move_step": int(self.preview_move_step.value()),
            "eyedropper_tol": int(self.eyedropper_tol.value()),
            "last_eyedrop_color": list(self.last_eyedrop_color),
            "show_raster": bool(self.chk_sprite.isChecked()),
            "show_skeleton": bool(self.chk_pose.isChecked()),
            "gen_strength": float(self.gen_strength.value()),
            "control_weight": float(self.control_weight.value()),
            "force_rig_warp": bool(self.force_rig_warp.isChecked()),
            "sdxl_base_model_path": self.sdxl_base_model_path.text().strip(),
            "topology_bones": self.topology_bones or [],
            "topology_new_node_counter": int(self.topology_new_node_counter),
            "frames": [self._frame_to_project_dict(fr, i) for i, fr in enumerate(self.frames)],
        }

        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.json", json.dumps(project, ensure_ascii=True, indent=2))
                for i, fr in enumerate(self.frames):
                    if fr.image is not None:
                        buf = io.BytesIO()
                        fr.image.save(buf, format="PNG")
                        zf.writestr(f"frames/{i}_image.png", buf.getvalue())
                    if fr.generated is not None:
                        buf = io.BytesIO()
                        fr.generated.save(buf, format="PNG")
                        zf.writestr(f"frames/{i}_generated.png", buf.getvalue())
            self.log(f"Progetto salvato: {path}")
        except Exception as e:
            self.log(f"Errore salvataggio progetto: {e}")
            QMessageBox.warning(self, "Errore", f"Salvataggio fallito: {e}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Importa progetto",
            str(Path.cwd()),
            "Animator Project (*.animproj)",
        )
        if not path:
            return

        try:
            with zipfile.ZipFile(path, "r") as zf:
                raw = zf.read("project.json")
                project = json.loads(raw.decode("utf-8"))

                frames_data = project.get("frames", [])
                count = int(project.get("frame_count", len(frames_data) or 0))
                count = max(2, count) if count else 8
                self.frames = [FrameData() for _ in range(count)]

                for fd in frames_data:
                    idx = int(fd.get("index", -1))
                    if idx < 0 or idx >= len(self.frames):
                        continue
                    fr = FrameData()
                    img_path = fd.get("image")
                    if img_path:
                        with zf.open(img_path) as f:
                            fr.image = Image.open(io.BytesIO(f.read())).convert("RGBA")
                            fr.image_unkeyed = fr.image.copy()
                    gen_path = fd.get("generated")
                    if gen_path:
                        with zf.open(gen_path) as f:
                            fr.generated = Image.open(io.BytesIO(f.read())).convert("RGBA")
                    fr.image_offset = tuple(fd.get("image_offset", [0, 0]))
                    rn = fd.get("rig_nodes") or {}
                    if rn:
                        fr.rig_nodes = {k: (float(v[0]), float(v[1])) for k, v in rn.items()}
                    fr.is_keyframe = bool(fd.get("is_keyframe", False))
                    fr.key_description = fd.get("key_description", "") or ""
                    fr.caption = fd.get("caption", "") or ""
                    fr.rig_profile = fd.get("rig_profile", "auto") or "auto"
                    fr.detected_profile = fd.get("detected_profile", "human") or "human"
                    self.frames[idx] = fr

                self.prompt.setPlainText(project.get("prompt", "") or "")
                self.negative.setPlainText(project.get("negative", "") or "")
                self.caption_hint.setPlainText(project.get("caption_hint", "") or "")
                self.use_gemini_caption.setChecked(bool(project.get("use_gemini_caption", True)))
                self.gemini_model.setText(project.get("gemini_model", "gemini-2.5-flash"))
                self.gen_strength.setValue(max(0.1, min(0.95, float(project.get("gen_strength", self.gen_strength.value())))))
                self.control_weight.setValue(max(0.2, min(2.5, float(project.get("control_weight", self.control_weight.value())))))
                self.force_rig_warp.setChecked(bool(project.get("force_rig_warp", self.force_rig_warp.isChecked())))
                self.sdxl_base_model_path.setText((project.get("sdxl_base_model_path", self.sdxl_base_model_path.text()) or "").strip())

                self.chk_onion.setChecked(bool(project.get("onion_enabled", False)))
                self.onion_range.setValue(int(project.get("onion_range", 2)))
                self.preview_move_step.setValue(int(project.get("move_step", 2)))
                self.eyedropper_tol.setValue(int(project.get("eyedropper_tol", 10)))
                col = project.get("last_eyedrop_color", [0, 0, 0])
                if isinstance(col, list) and len(col) == 3:
                    self.last_eyedrop_color = (int(col[0]), int(col[1]), int(col[2]))
                self._update_eyedropper_chip()

                self.chk_sprite.setChecked(bool(project.get("show_raster", True)))
                self.chk_pose.setChecked(bool(project.get("show_skeleton", True)))

                raw_topology = project.get("topology_bones") or None
                if isinstance(raw_topology, list):
                    norm_topology: List[Tuple[str, str]] = []
                    for b in raw_topology:
                        if isinstance(b, (list, tuple)) and len(b) == 2:
                            norm_topology.append((str(b[0]), str(b[1])))
                    self.topology_bones = norm_topology or None
                else:
                    self.topology_bones = None
                self.topology_new_node_counter = int(project.get("topology_new_node_counter", 1))

                # Recompute bone lengths for loaded frames.
                for i, fr in enumerate(self.frames):
                    if fr.rig_nodes is not None:
                        bones = self.get_frame_bones(fr)
                        fr.bone_lengths = compute_bone_lengths(fr.rig_nodes, bones)
                        self.frames[i] = fr

                self.selected_index = int(project.get("selected_index", 0))
                self.selected_index = max(0, min(self.selected_index, len(self.frames) - 1))
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(len(self.frames))
                self.frame_slider.blockSignals(False)

                self.playing = False
                self.play_timer.stop()
                if hasattr(self, "btn_play"):
                    self.btn_play.blockSignals(True)
                    self.btn_play.setChecked(False)
                    self.btn_play.setText("play")
                    self.btn_play.blockSignals(False)

                self.rig_approved = False
                self.btn_generate_frames.setEnabled(True)

                self.refresh_ui()
                self.log(f"Progetto importato: {path}")
        except Exception as e:
            self.log(f"Errore importazione progetto: {e}")
            QMessageBox.warning(self, "Errore", f"Importazione fallita: {e}")

    def pick_sdxl_model_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona checkpoint SDXL",
            str(Path.cwd()),
            "Model files (*.safetensors *.ckpt)",
        )
        if not path:
            return
        self.sdxl_base_model_path.setText(path)
        self.log(f"[raster] Checkpoint SDXL selezionato: {path}")

    def refresh_ui(self):
        self.rebuild_thumbs()
        self.refresh_preview()
        self.update_frame_label()
        self.sync_keyframe_editor()
        self.btn_generate_frames.setEnabled(True)

    def mark_rig_dirty(self):
        return

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
            fr.image_unkeyed = fr.image.copy()
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

        if fr.image_unkeyed is None and fr.image is not None:
            fr.image_unkeyed = fr.image.convert("RGBA")
        src = (fr.image_unkeyed or fr.image).convert("RGBA")
        r, g, b, a = src.getpixel((sx, sy))
        if a == 0:
            self.log("Contagocce: pixel gia trasparente, nessuna modifica")
            return

        # Re-apply from unkeyed source so the previous key color is replaced, not accumulated.
        fr.image = apply_chroma_key(src, (r, g, b), tol)
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

    def toggle_play(self, enabled: bool):
        self.playing = bool(enabled)
        if self.playing:
            if len(self.frames) < 2:
                self.playing = False
                self.btn_play.blockSignals(True)
                self.btn_play.setChecked(False)
                self.btn_play.blockSignals(False)
                self.log("Play: servono almeno 2 frame")
                return
            self.btn_play.setText("stop")
            self.play_timer.start()
        else:
            self.btn_play.setText("play")
            self.play_timer.stop()

    def on_play_tick(self):
        if not self.playing:
            return
        if len(self.frames) < 2:
            self.play_timer.stop()
            self.playing = False
            self.btn_play.setChecked(False)
            self.btn_play.setText("play")
            return
        next_idx = (self.selected_index + 1) % len(self.frames)
        self.set_selected_frame(next_idx)

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
        fr.image_unkeyed = img.copy()

        if key_enabled and len(key_color) == 7 and key_color.startswith("#"):
            kr = int(key_color[1:3], 16)
            kg = int(key_color[3:5], 16)
            kb = int(key_color[5:7], 16)
            img = apply_chroma_key(fr.image_unkeyed, (kr, kg, kb), key_tol)

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
            norm_topology: List[Tuple[str, str]] = []
            for b in self.topology_bones:
                if isinstance(b, (list, tuple)) and len(b) == 2:
                    norm_topology.append((str(b[0]), str(b[1])))
            self.topology_bones = norm_topology
            return self.topology_bones
        return RIG_BONES_BY_PROFILE.get(self.resolve_rig_profile(fr), HUMAN_BONES)

    def analyze_frame_bone_correspondence(self, idx: int, log_result: bool = False) -> Dict[Tuple[str, str], str]:
        if idx < 0 or idx >= len(self.frames):
            return {}
        fr = self.frames[idx]
        if fr.rig_nodes is None:
            return {}
        bones = self.get_frame_bones(fr)
        if not bones:
            return {}
        ref_img = fr.image or fr.generated or self.frames[0].image
        if ref_img is None:
            return {}
        canvas_size = self.get_frame_canvas_size(idx) or ref_img.size
        mask = build_subject_mask(ref_img)

        profile = self.resolve_rig_profile(fr)
        template_bones = RIG_BONES_BY_PROFILE.get(profile, HUMAN_BONES)
        template_nodes = self.create_default_rig(ref_img, fr)
        if template_nodes is None:
            mapped = {b: f"{b[0]}->{b[1]}" for b in bones}
            fr.bone_correspondence = mapped
            self.frames[idx] = fr
            return mapped

        mapped = infer_bone_correspondence(
            fr.rig_nodes,
            bones,
            template_nodes,
            template_bones,
            canvas_size,
            mask,
        )
        fr.bone_correspondence = mapped
        self.frames[idx] = fr
        if log_result:
            sample = ", ".join([f"{a}->{b}:{role}" for (a, b), role in list(mapped.items())[:8]])
            self.log(f"[debug/bone-map] frame {idx + 1}: segmenti={len(mapped)} sample={sample}")
        return mapped

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
        self.frames[idx].image_unkeyed = img.copy()
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

    def _log_gemini_motion_debug(self, key_idxs: List[int], explicit_keys: List[int]):
        prompt_text = (self.prompt.toPlainText() or "").strip()
        prompt_preview = prompt_text[:180] + ("..." if len(prompt_text) > 180 else "")
        has_api = bool(self.gemini_api_key.text().strip())
        model_name = self.gemini_model.text().strip() or "gemini-2.5-flash"
        frames_with_rig = [i + 1 for i, fr in enumerate(self.frames) if fr.rig_nodes is not None]
        frame_profiles = [self.resolve_rig_profile(fr) for fr in self.frames]

        self.log("[debug/gemini-motion] ----")
        self.log(f"[debug/gemini-motion] use_gemini_caption={self.use_gemini_caption.isChecked()} api_key_set={has_api}")
        self.log(f"[debug/gemini-motion] model={model_name}")
        self.log(f"[debug/gemini-motion] frames={len(self.frames)} frames_with_rig={frames_with_rig}")
        self.log(f"[debug/gemini-motion] explicit_keyframes={[i + 1 for i in explicit_keys]} anchors={[i + 1 for i in key_idxs]}")
        self.log(f"[debug/gemini-motion] rig_profiles={frame_profiles}")
        self.log(f"[debug/gemini-motion] prompt='{prompt_preview or '(vuoto)'}'")
        if not has_api:
            self.log("[debug/gemini-motion] API key Gemini assente -> fallback locale")
        elif not self.use_gemini_caption.isChecked():
            self.log("[debug/gemini-motion] Gemini disattivato -> fallback locale")
        else:
            self.log("[debug/gemini-motion] API key presente -> fallback locale (motion-plan remoto non attivo)")

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        import re

        raw = (text or "").strip()
        if not raw:
            return ""
        fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, flags=re.IGNORECASE)
        if fence:
            return fence.group(1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1].strip()
        return raw

    def _request_gemini_motion_plan(
        self,
        key_idxs: List[int],
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, Tuple[float, float]]]]:
        if not self.use_gemini_caption.isChecked():
            return {}, {}
        api_key = normalize_gemini_api_key(self.gemini_api_key.text())
        if not api_key:
            return {}, {}

        anchor_idx = key_idxs[0]
        anchor = self.frames[anchor_idx]
        if anchor.rig_nodes is None:
            return {}, {}
        bones = self.get_frame_bones(anchor)
        correspondence = anchor.bone_correspondence or self.analyze_frame_bone_correspondence(anchor_idx, log_result=False)
        node_names = sorted(anchor.rig_nodes.keys())
        frame_size = self.get_frame_canvas_size(anchor_idx)
        if frame_size is None:
            return {}, {}
        fw, fh = frame_size
        if fw <= 0 or fh <= 0:
            return {}, {}

        anchor_norm = {
            n: [round(anchor.rig_nodes[n][0] / fw, 4), round(anchor.rig_nodes[n][1] / fh, 4)] for n in node_names
        }
        key_desc = []
        for k in key_idxs:
            fr = self.frames[k]
            key_desc.append(
                {
                    "frame": k,
                    "description": (fr.key_description or fr.caption or "").strip(),
                }
            )

        prompt = {
            "task": "Generate coherent skeletal motion plan for 2D sprite animation.",
            "language": "it",
            "user_animation_prompt": (self.prompt.toPlainText() or "").strip(),
            "frames_count": len(self.frames),
            "rig_profile": self.resolve_rig_profile(anchor),
            "nodes": node_names,
            "bones": bones,
            "bone_to_anatomy_map": [{"bone": [a, b], "anatomy": correspondence.get((a, b), f"{a}->{b}")} for (a, b) in bones],
            "anchor_frame_index": anchor_idx,
            "anchor_nodes_normalized_xy": anchor_norm,
            "keyframes_hint": key_desc,
            "constraints": [
                "Return ONLY JSON object, no markdown.",
                "Use normalized offsets relative to width/height.",
                "Keep offsets in range [-0.25, 0.25].",
                "Prefer smooth periodic motion with temporal coherence.",
                "Keep torso/head relatively stable and move distal limbs more.",
            ],
            "output_schema": {
                "frames": [
                    {
                        "index": 0,
                        "global_offset": [0.0, 0.0],
                        "node_offsets": {"node_name": [0.0, 0.0]},
                    }
                ]
            },
        }

        model = self.gemini_model.text().strip() or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=True)}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        try:
            res = requests.post(
                url,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                params={"key": api_key},
                json=body,
                timeout=60,
            )
            if not res.ok:
                self.log(f"[debug/gemini-motion] {format_gemini_http_error(res)} -> fallback locale")
                return {}, {}
            payload = res.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()
            if not text:
                self.log("[debug/gemini-motion] Gemini risposta vuota: fallback locale")
                return {}, {}
            candidate = self._extract_json_candidate(text)
            data = json.loads(candidate)
        except Exception as e:
            self.log(f"[debug/gemini-motion] parsing/chiamata Gemini fallita: {e}")
            return {}, {}

        frames = data.get("frames", []) if isinstance(data, dict) else []
        if not isinstance(frames, list):
            self.log("[debug/gemini-motion] schema non valido (frames assente): fallback locale")
            return {}, {}

        global_offsets: Dict[int, Tuple[float, float]] = {}
        node_offsets: Dict[int, Dict[str, Tuple[float, float]]] = {}
        valid_entries = 0
        for item in frames:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(self.frames):
                continue
            g = item.get("global_offset", [0.0, 0.0])
            if isinstance(g, list) and len(g) == 2:
                try:
                    gx = max(-0.25, min(0.25, float(g[0])))
                    gy = max(-0.25, min(0.25, float(g[1])))
                    global_offsets[idx] = (gx, gy)
                except Exception:
                    pass

            nmap: Dict[str, Tuple[float, float]] = {}
            no = item.get("node_offsets", {})
            if isinstance(no, dict):
                for n, vec in no.items():
                    if n not in node_names or not isinstance(vec, list) or len(vec) != 2:
                        continue
                    try:
                        ox = max(-0.25, min(0.25, float(vec[0])))
                        oy = max(-0.25, min(0.25, float(vec[1])))
                        nmap[n] = (ox, oy)
                    except Exception:
                        continue
            node_offsets[idx] = nmap
            valid_entries += 1

        if valid_entries == 0:
            self.log("[debug/gemini-motion] nessun frame valido nel motion-plan: fallback locale")
            return {}, {}
        self.log(f"[debug/gemini-motion] motion-plan Gemini ok: frames_validi={valid_entries} node_set={len(node_names)}")
        return global_offsets, node_offsets

    @staticmethod
    def _clip_nodes_to_canvas(nodes: Dict[str, Tuple[float, float]], size: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
        w, h = size
        out = {}
        for n, (x, y) in nodes.items():
            out[n] = (
                max(0.0, min(float(w - 1), float(x))),
                max(0.0, min(float(h - 1), float(y))),
            )
        return out

    def simulate_rig_animation(self):
        explicit_keys = [i for i, fr in enumerate(self.frames) if fr.is_keyframe and fr.rig_nodes is not None]
        key_idxs = list(explicit_keys)
        if not key_idxs:
            key_idxs = [i for i, fr in enumerate(self.frames) if fr.rig_nodes is not None]
        if not key_idxs:
            QMessageBox.warning(self, "Attenzione", "Carica almeno un frame chiave")
            return

        for k in key_idxs:
            self.analyze_frame_bone_correspondence(k, log_result=True)
        self._log_gemini_motion_debug(key_idxs, explicit_keys)
        gemini_global_offsets, gemini_node_offsets = self._request_gemini_motion_plan(key_idxs)
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
            if fr.rig_nodes is not None and (i in gemini_global_offsets or i in gemini_node_offsets):
                frame_size = self.get_frame_canvas_size(i)
                if frame_size is not None:
                    gw = float(frame_size[0])
                    gh = float(frame_size[1])
                    gx, gy = gemini_global_offsets.get(i, (0.0, 0.0))
                    global_px = (gx * gw * 0.55, gy * gh * 0.55)
                    per_node = gemini_node_offsets.get(i, {})
                    adjusted = {}
                    for n, (x, y) in fr.rig_nodes.items():
                        ox, oy = per_node.get(n, (0.0, 0.0))
                        adjusted[n] = (
                            x + global_px[0] + ox * gw * 0.85,
                            y + global_px[1] + oy * gh * 0.85,
                        )
                    fr.rig_nodes = self._clip_nodes_to_canvas(adjusted, frame_size)
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
        idx0 = next((i for i, fr in enumerate(self.frames) if fr.image is not None), None)
        if idx0 is None:
            QMessageBox.warning(self, "Attenzione", "Carica almeno un frame con sprite")
            return

        for i, fr in enumerate(self.frames):
            if fr.rig_nodes is None:
                QMessageBox.warning(self, "Attenzione", f"Manca il rig nel frame {i + 1}")
                return

        use_cuda = torch.cuda.is_available()
        model_choice = (self.sdxl_base_model_path.text() or "").strip()
        base_model_id = model_choice if model_choice else SDXL_BASE_MODEL_ID
        base_model_path = Path(base_model_id)
        if base_model_id.lower().endswith((".safetensors", ".ckpt")) and not base_model_path.exists():
            QMessageBox.warning(self, "Modello SDXL", f"File checkpoint non trovato:\n{base_model_id}")
            return
        try:
            pipe = load_pipe(base_model_id, SDXL_CONTROLNET_OPENPOSE_ID, use_cuda)
            pipe_i2i = load_pipe_img2img(base_model_id, SDXL_CONTROLNET_OPENPOSE_ID, use_cuda)
            self.log(f"[raster] Backend diffusion: SDXL ({base_model_id}) + ControlNet ({SDXL_CONTROLNET_OPENPOSE_ID})")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Caricamento modelli fallito: {e}")
            return

        base_rgba = self.frames[idx0].image.convert("RGBA")
        base_img = base_rgba.convert("RGB")
        base_nodes_ref = self.frames[idx0].rig_nodes or {}
        plan = calc_gen_size(base_img.width, base_img.height, "resize_down")
        prompt_txt = self.prompt.toPlainText().strip()
        if not prompt_txt:
            prompt_txt = (self.frames[idx0].caption or "").strip()
        if not prompt_txt:
            prompt_txt = "pixel art game sprite, clean silhouette, consistent character design"
        neg_txt = self.negative.toPlainText().strip()
        if not neg_txt:
            neg_txt = "blurry, noisy, glitch, abstract, deformed, bad anatomy, text, watermark, low quality"
        strength_val = max(0.1, min(0.95, float(self.gen_strength.value())))
        control_val = max(0.2, min(2.5, float(self.control_weight.value())))
        use_rig_warp_mode = bool(self.force_rig_warp.isChecked()) or max(base_rgba.size) <= 192
        if use_rig_warp_mode:
            self.log("[raster] Modalita rig-warp attiva (sprite piccolo): bypass diffusion")

        generator = torch.Generator(device="cuda" if use_cuda else "cpu").manual_seed(42)
        prev_raster: Optional[Image.Image] = None
        prev_raster_work: Optional[Image.Image] = None
        prev_nodes: Optional[Dict[str, Tuple[float, float]]] = None

        for i, fr in enumerate(self.frames):
            try:
                if fr.image is not None:
                    # Keep explicitly provided raster frames untouched.
                    src_rgba = compose_with_offset(fr.image, fr.image_offset).convert("RGBA")
                    fr.generated = src_rgba
                    prev_raster = fr.generated
                    prev_raster_work = src_rgba
                    prev_nodes = dict(fr.rig_nodes) if fr.rig_nodes is not None else None
                    self.log(f"Frame {i + 1}: raster caricato, salto rigenerazione")
                    continue

                corr_map = fr.bone_correspondence or self.analyze_frame_bone_correspondence(i, log_result=False)
                anatomy_terms = sorted(set(corr_map.values())) if corr_map else []
                anatomy_hint = ", ".join(anatomy_terms[:10])
                prompt_frame = prompt_txt if not anatomy_hint else f"{prompt_txt}. Skeleton anatomy map: {anatomy_hint}."
                self.log(
                    f"[raster] frame {i + 1}: bone-map {'presente' if corr_map else 'assente'}"
                    + (f" ({len(corr_map)} segmenti)" if corr_map else "")
                )

                if use_rig_warp_mode and fr.rig_nodes is not None and base_nodes_ref:
                    warp_src = base_rgba
                    warp_nodes = base_nodes_ref
                    mode = "rig_warp_base"
                    if prev_raster_work is not None and prev_nodes is not None:
                        warp_src = prev_raster_work
                        warp_nodes = prev_nodes
                        mode = "rig_warp_chain"
                    # Keep an unfiltered working raster for the next step to avoid cumulative erosion.
                    out_work = warp_sprite_with_rig(warp_src, warp_nodes, fr.rig_nodes)
                    out_rgba = quantize_to_reference_palette(out_work, base_rgba, max_colors=48)
                    tol = int(self.eyedropper_tol.value())
                    key_rgb = pick_effective_key_color(out_rgba, self.last_eyedrop_color, tol)
                    out_rgba = apply_chroma_key(out_rgba, key_rgb, tol, edge_connected=True)
                    fr.generated = out_rgba
                    prev_raster = out_rgba
                    prev_raster_work = out_work
                    prev_nodes = dict(fr.rig_nodes)
                    self.log(f"Generato frame {i + 1}/{len(self.frames)} mode={mode}")
                    continue

                bones = self.get_frame_bones(fr)
                frame_size = self.get_frame_canvas_size(i) or base_img.size
                pose_map = rig_to_pose_map(fr.rig_nodes, frame_size, bones)
                pose_in = resize_or_pad(pose_map.convert("RGB"), plan)
                motion_score = 0.0
                strength_eff = strength_val
                control_eff = control_val
                guidance_eff = 5.6
                use_i2i = prev_raster is not None
                mode = "img2img_chain"
                if prev_raster is not None:
                    init_src = prev_raster.convert("RGBA")
                    if prev_nodes is not None and fr.rig_nodes is not None:
                        dx, dy = estimate_global_rig_shift(prev_nodes, fr.rig_nodes)
                        dx = int(round(dx * 1.60))
                        dy = int(round(dy * 1.60))
                        init_src = shift_image_on_canvas(init_src, dx, dy)
                        motion_score = rig_motion_score(prev_nodes, fr.rig_nodes, frame_size)
                        strength_eff = max(0.20, min(0.98, strength_val + motion_score * 0.55))
                        control_eff = max(0.6, min(2.5, control_val + motion_score * 1.00))
                        guidance_eff = max(4.8, min(6.2, 5.8 - motion_score * 0.8))
                        # Keep chain behavior: always use previous frame as reference.
                        use_i2i = True
                    if use_i2i:
                        init_prev = resize_or_pad(init_src.convert("RGB"), plan)
                        init_img = init_prev
                        mode = "img2img_chain"
                        out = pipe_i2i(
                            prompt=prompt_frame,
                            negative_prompt=neg_txt,
                            image=init_img,
                            control_image=pose_in,
                            strength=strength_eff,
                            num_inference_steps=28,
                            guidance_scale=guidance_eff,
                            controlnet_conditioning_scale=control_eff,
                            generator=generator,
                        ).images[0]
                    else:
                        mode = "txt2img_pose_lock"
                        out = pipe(
                            prompt=prompt_frame,
                            negative_prompt=neg_txt,
                            image=pose_in,
                            num_inference_steps=34,
                            guidance_scale=max(5.8, guidance_eff),
                            controlnet_conditioning_scale=max(control_eff, control_val + 0.5),
                            generator=generator,
                            width=plan.gen_w,
                            height=plan.gen_h,
                        ).images[0]
                else:
                    mode = "txt2img_init"
                    out = pipe(
                        prompt=prompt_frame,
                        negative_prompt=neg_txt,
                        image=pose_in,
                        num_inference_steps=32,
                        guidance_scale=6.0,
                        controlnet_conditioning_scale=control_eff,
                        generator=generator,
                        width=plan.gen_w,
                        height=plan.gen_h,
                    ).images[0]
                out = crop_or_resize_back(out, plan)
                out_rgba = out.convert("RGBA")
                # Keep generated frames in the same color family as the source sprite.
                out_rgba = quantize_to_reference_palette(out_rgba, base_rgba, max_colors=48)
                # Apply chroma key chosen with eyedropper instead of forcing an external alpha mask.
                tol = int(self.eyedropper_tol.value())
                key_rgb = pick_effective_key_color(out_rgba, self.last_eyedrop_color, tol)
                out_rgba = apply_chroma_key(out_rgba, key_rgb, tol, edge_connected=True)
                fr.generated = out_rgba
                prev_raster = out_rgba
                prev_nodes = dict(fr.rig_nodes) if fr.rig_nodes is not None else None
                self.log(
                    f"Generato frame {i + 1}/{len(self.frames)} mode={mode} (motion={motion_score:.2f}, strength={strength_eff:.2f}, control={control_eff:.2f})"
                )
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




