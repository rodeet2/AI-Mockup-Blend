import os
import json
import base64
import io
import numpy as np
import cv2
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DPTImageProcessor, AutoModelForDepthEstimation

# Force transformers to use only local cache — no network calls
os.environ["TRANSFORMERS_OFFLINE"] = "1"

app = Flask(__name__)
CORS(app)

# ── Load Depth Anything once at startup ────────────────────────────────────
CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf"
    "/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6"
)
print(f"Loading Depth Anything from cache…")
depth_processor = DPTImageProcessor.from_pretrained(CACHE_PATH)
depth_model     = AutoModelForDepthEstimation.from_pretrained(CACHE_PATH)
depth_model.eval()
print("Depth model ready.")


def estimate_depth(pil_rgb: Image.Image) -> np.ndarray:
    """Return a float32 depth array (H×W) for the given RGB PIL image."""
    inputs = depth_processor(images=pil_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
    depth = outputs.predicted_depth  # (1, H, W)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(pil_rgb.height, pil_rgb.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().numpy()
    return depth.astype(np.float32)


@app.route("/merge", methods=["POST"])
def merge():
    # ── Validate inputs ────────────────────────────────────────────────
    if "product" not in request.files or "design" not in request.files:
        return jsonify({"error": "Missing product or design file."}), 400

    placement_raw = request.form.get("placement")
    if not placement_raw:
        return jsonify({"error": "Missing placement."}), 400

    try:
        p = json.loads(placement_raw)
        x, y, w, h = int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])
        assert w > 0 and h > 0
    except Exception:
        return jsonify({"error": "placement must be JSON with x, y, w, h."}), 400

    displacement_strength = float(request.form.get("strength", "20"))
    blend_amount          = float(request.form.get("blend", "0.85"))
    blend_mode            = request.form.get("blend_mode", "overlay")

    # ── Load images ────────────────────────────────────────────────────
    try:
        product_img = Image.open(request.files["product"]).convert("RGBA")
        design_img  = Image.open(request.files["design"]).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    pw, ph = product_img.size

    # Clamp placement to product bounds
    x = max(0, min(x, pw - 1))
    y = max(0, min(y, ph - 1))
    w = min(w, pw - x)
    h = min(h, ph - y)

    # ── Crop product area for depth estimation ─────────────────────────
    crop_pil = product_img.crop((x, y, x + w, y + h)).convert("RGB")

    # ── Run depth model on the crop ────────────────────────────────────
    depth_map = estimate_depth(crop_pil)             # (h_crop, w_crop)
    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

    # Smooth before computing gradients (removes JPEG noise)
    depth_smooth = cv2.GaussianBlur(depth_norm, (0, 0), sigmaX=3)

    # ── Build displacement field from 3D surface shape ─────────────────
    # Center the depth map at zero: raised areas are positive, recessed
    # areas are negative. This drives smooth wrapping of the full surface
    # contour rather than only reacting at sharp fold edges (Sobel).
    depth_centered = depth_smooth - depth_smooth.mean()
    max_abs = max(np.abs(depth_centered).max(), 1e-6)
    depth_displaced = depth_centered / max_abs  # -1.0 to +1.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + depth_displaced * displacement_strength
    map_y = yy + depth_displaced * displacement_strength

    # ── Resize design and apply displacement warp ──────────────────────
    design_resized = design_img.resize((w, h), Image.LANCZOS)
    design_cv = cv2.cvtColor(np.array(design_resized), cv2.COLOR_RGBA2BGRA)

    warped = cv2.remap(
        design_cv, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # ── Blend design with product surface lighting ─────────────────────
    warped_f = warped.astype(np.float32) / 255.0

    if blend_amount > 0 and blend_mode != 'source-over':
        crop_np = np.array(crop_pil).astype(np.float32) / 255.0
        ch_map  = [2, 1, 0]  # warped BGR index → crop RGB index

        for c, pc in enumerate(ch_map):
            src = crop_np[:, :, pc]
            dst = warped_f[:, :, c]

            if blend_mode == 'overlay':
                blended = np.where(
                    src < 0.5,
                    2 * src * dst,
                    1 - 2 * (1 - src) * (1 - dst)
                )
            elif blend_mode == 'soft-light':
                blended = (1 - 2 * src) * dst ** 2 + 2 * src * dst
            elif blend_mode == 'multiply':
                blended = src * dst
            elif blend_mode == 'screen':
                blended = 1 - (1 - src) * (1 - dst)
            else:
                blended = dst

            warped_f[:, :, c] = dst * (1 - blend_amount) + blended * blend_amount

        warped_f[:, :, :3] = np.clip(warped_f[:, :, :3], 0, 1)

    # ── Apply opacity to alpha channel — matches canvas globalAlpha ────
    warped_f[:, :, 3] = np.clip(warped_f[:, :, 3] * blend_amount, 0, 1)

    warped = (warped_f * 255).astype(np.uint8)
    warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA))

    # ── Composite: product untouched as base, design on top ───────────
    result = product_img.copy()
    result.alpha_composite(warped_pil, dest=(x, y))

    # ── Encode as base64 PNG ───────────────────────────────────────────
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({"result": encoded})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
