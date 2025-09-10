from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import io, os, uuid, time, sys, re
import json

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ============================================================
# INFERENCE-ONLY SERVER
#  - Loads pre-trained checkpoints (.pt) + classes.txt files
#  - NO TRAINING happens here
# ============================================================

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parents[1]      # project root
TRAINING = ROOT / "training"                     # training assets
CKPT_DIR = TRAINING / "checkpoints"

# General classifier (required)
CKPT = CKPT_DIR / "best_efficientnet_v2_s.pt"
CLASSES_TXT = CKPT_DIR / "classes.txt"

# Apple-health classifier (optional)
APPLE_DIR = CKPT_DIR / "apple_health"
APPLE_CKPT = APPLE_DIR / "model_mc_best.pth"
APPLE_CLASSES_TXT = APPLE_DIR / "classes.txt"
# YOLO detector (optional but recommended)
YOLO_DIR = CKPT_DIR / "yolo"
YOLO_CKPT = YOLO_DIR / "best.pt"
YOLO_CLASSES_TXT = YOLO_DIR / "classes.txt"#??

# Annotations destination (optional feature)
DATA_DIR = TRAINING / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ensure THIS repo’s root is first on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.model import build_model  # import explicitly from your package


# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Transforms / Config --------------------
IMG_SIZE = 160
IMG_SIZE_APPLE = 160
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TFM = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
TFM_APPLE = T.Compose([T.Resize(IMG_SIZE_APPLE), T.CenterCrop(IMG_SIZE_APPLE), T.ToTensor(), T.Normalize(MEAN, STD)])

CLS_THRESH = 0.50     # UI hint only
YOLO_CONF = 0.25
YOLO_IOU = 0.45
PADDING = 6
# if YOLO and general-class names differ, map them here
CLASS_MAP: Dict[str, str] = {}

# -------------------- Utils --------------------
def _load_classes(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def _sanitize_label(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)

# -------------------- Load General Classifier (REQUIRED) --------------------
if not CKPT.exists():
    raise FileNotFoundError(f"[INFERENCE] Missing general checkpoint: {CKPT}")
if not CLASSES_TXT.exists():
    raise FileNotFoundError(f"[INFERENCE] Missing general classes.txt: {CLASSES_TXT}")

CLASSES = _load_classes(CLASSES_TXT)
general_model = build_model(num_classes=len(CLASSES), dropout=0.3, pretrained=False).to(device)

state = torch.load(CKPT, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model" in state:
    state = state["model"]
general_model.load_state_dict(state, strict=False)
general_model.eval()

# -------------------- Load Apple-health Classifier (OPTIONAL) --------------------
apple_model: Optional[nn.Module] = None
APPLE_CLASSES: Optional[List[str]] = None

if APPLE_CKPT.exists() and APPLE_CLASSES_TXT.exists():
    APPLE_CLASSES = _load_classes(APPLE_CLASSES_TXT)
    apple_model = build_model(num_classes=len(APPLE_CLASSES), dropout=0.3, pretrained=False).to(device)
    a_state = torch.load(APPLE_CKPT, map_location=device)
    if isinstance(a_state, dict) and "state_dict" in a_state:
        a_state = a_state["state_dict"]
    elif isinstance(a_state, dict) and "model" in a_state:
        a_state = a_state["model"]
    apple_model.load_state_dict(a_state, strict=False)
    apple_model.eval()
else:
    print("[INFERENCE][WARN] Apple health classifier not found. 'apple_health' will be null.")

# -------------------- Load YOLO (OPTIONAL) --------------------
try:
    from ultralytics import YOLO
    _ULTRA_OK = True
except Exception as e:
    print("[INFERENCE][WARN] ultralytics not installed or failed to import:", e)
    _ULTRA_OK = False

yolo_model = None
if _ULTRA_OK and YOLO_CKPT.exists():
    yolo_model = YOLO(str(YOLO_CKPT))
else:
    print("[INFERENCE][WARN] YOLO weights not found. 'yolo' list will be empty.")

# ---- NEW: get YOLO class names (from file or from model) ----
YOLO_NAMES: Dict[int, str] = {}
YOLO_KNOWN: set[str] = set()
try:
    if YOLO_CLASSES_TXT.exists():
        names = [ln.strip() for ln in YOLO_CLASSES_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
        YOLO_NAMES = {i: n for i, n in enumerate(names)}
    elif yolo_model is not None:
        # fallback: read embedded names
        YOLO_NAMES = getattr(yolo_model, "model", yolo_model).names  # dict {id: name}
    YOLO_KNOWN = set(YOLO_NAMES.values())
    print("[YOLO] Known classes:", YOLO_KNOWN)
except Exception as e:
    print("[YOLO] Could not determine YOLO class names:", e)

# -------------------- FastAPI --------------------
app = FastAPI(title="Moptimizer API — Inference Only")
from fastapi.middleware.cors import CORSMiddleware
import os

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5175",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:5173",
    "http://localhost:5176",
    "http://127.0.0.1:5176",
     "http://localhost:5177",
     "http://127.0.0.1:5177",
]

extra = os.getenv("UI_ORIGINS")
if extra:
    ALLOWED_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # use ["*"] temporarily while debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "mode": "inference-only"}

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": str(device),
        "img_size": IMG_SIZE,
        "num_classes": len(CLASSES),
        "apple_health_available": bool(apple_model is not None),
        "yolo_available": bool(yolo_model is not None),
    }

@app.get("/classes")
def classes():
    return {"classes": CLASSES}

# -------------------- Inference helpers --------------------
@torch.no_grad()
def _predict_general_pil(img: Image.Image) -> Tuple[str, List[float], int, float]:
    x = TFM(img.convert("RGB")).unsqueeze(0).to(device)
    logits = general_model(x)
    probs = torch.softmax(logits, dim=1)[0]
    score, idx = torch.max(probs, dim=0)
    label = CLASSES[int(idx)]
    return label, probs.cpu().tolist(), int(idx), float(score.item())

@torch.no_grad()
def _predict_apple_health_pil(img: Image.Image) -> Optional[Tuple[str, float]]:
    if apple_model is None or APPLE_CLASSES is None:
        return None
    x = TFM_APPLE(img.convert("RGB")).unsqueeze(0).to(device)
    logits = apple_model(x)
    probs = torch.softmax(logits, dim=1)[0]
    score, idx = torch.max(probs, dim=0)
    label = APPLE_CLASSES[int(idx)]
    return label, float(score.item())

@torch.no_grad()
def _run_yolo(img: Image.Image, conf: float = YOLO_CONF, iou: float = YOLO_IOU):
    """
    Returns detections:
    [{label, score, box:[x1,y1,x2,y2], box_norm:[x1,y1,x2,y2]}]
    'box' is in absolute pixels; 'box_norm' is normalized by width/height.
    """
    if yolo_model is None:
        return []
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    results = yolo_model.predict(source=img, conf=conf, iou=iou, verbose=False, device=device_str)
    dets = []
    W, H = img.size
    for r in results:
        boxes = r.boxes
        names = r.names
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()
        for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
            label = names.get(int(cls_id), str(int(cls_id)))
            dets.append({
                "label": label,
                "score": float(c),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "box_norm": [float(x1)/W, float(y1)/H, float(x2)/W, float(y2)/H],
            })
    return dets

#keep that we don't do out oof bounds
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# -------------------- Single endpoint: full pipeline --------------------

@app.post("/predict_pipeline")
async def predict_pipeline(file: UploadFile = File(description="Upload an image file for prediction")):
    """
     1) General classification → label
     2) YOLO → get boxes (filtered by the general class if רלוונטי)
     3) For each YOLO box → crop → run apple_health classifier per-object
    """
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    W, H = img.size

    # 1) general classification
    cls_label, probs, idx, cls_score = _predict_general_pil(img)

   

    # 3) YOLO filtered by class from step 1
    yolo_label = CLASS_MAP.get(cls_label, cls_label)
    all_dets = _run_yolo(img, conf=YOLO_CONF, iou=YOLO_IOU)
    
    if yolo_label in YOLO_KNOWN:
        dets = [d for d in all_dets if d["label"] == yolo_label]

    else:
        # If the classifier label isn't a YOLO class, show all YOLO detections
        # (so the array isn't empty and you still see what YOLO found).
        dets = all_dets
    
    for d in dets:
        try:
            x1, y1, x2, y2 = d["box"]
            #padding + clamp
            x1 = _clamp(int(x1) - PADDING, 0, W - 1)
            y1 = _clamp(int(y1) - PADDING, 0, H - 1)
            x2 = _clamp(int(x2) + PADDING, 1, W)
            y2 = _clamp(int(y2) + PADDING, 1, H)

            if d["label"] != None:
                crop = img.crop((x1,y1,x2,y2))
                ah = _predict_apple_health_pil(crop)  # (label, score) or None
                if ah is not None:
                    ah_label, ah_score = ah
                    d["apple_health"] = {"label": ah_label, "score": ah_score}

        except Exception as e:
            print("[PIPELINE] Could not run apple_health on box:", e)
    
    
    return {
        "classification": {"label": cls_label, "score": cls_score, "index": idx},
        "yolo": dets,                   # only boxes for the chosen class
        "display_boxes_for": yolo_label,
        "meta": {"cls_thresh": CLS_THRESH, "yolo_conf": YOLO_CONF, "yolo_iou": YOLO_IOU},
    }

# -------------------- Annotations (optional helper) --------------------
@app.post("/annotate")
async def annotate(file: UploadFile = File(...), label: str = Form(...)):
    if not label or not label.strip():
        return JSONResponse({"ok": False, "error": "label required"}, status_code=400)

    safe_label = _sanitize_label(label)
    if not safe_label:
        return JSONResponse({"ok": False, "error": "invalid label"}, status_code=400)

    raw = await file.read()
    try:
        Image.open(io.BytesIO(raw))
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid image"}, status_code=400)

    subdir = DATA_DIR / safe_label
    subdir.mkdir(parents=True, exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        ext = ".png"

    fname = f"{int(time.time())}_{uuid.uuid4().hex}{ext}"
    out_path = subdir / fname
    out_path.write_bytes(raw)

    try:
        import csv
        log_path = DATA_DIR / "annotations_log.csv"
        new_file = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "label", "path", "original_name"])
            w.writerow([int(time.time()), safe_label, str(out_path), file.filename or ""])
    except Exception:
        pass

    return {"ok": True, "saved_to": str(out_path), "label": safe_label}
