#!/usr/bin/env python3
"""
Export per-frame binary masks with SAM-I2V for Gaussian Grouping.

Writes uint8 PNGs with id 0 (background) and id 1 (target object) into
<output>/<frame_basename>.png aligned with the input frame order.
"""
import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
from contextlib import nullcontext


def sort_frame_names(frame_names: List[str]) -> List[str]:
    """Sort frame names by the first integer in the basename (matches SAM-I2V loader)."""

    def extract_number(filename: str) -> int:
        name, _ = os.path.splitext(filename)
        match = re.search(r"\d+", name)
        return int(match.group()) if match else -1

    return sorted(frame_names, key=extract_number)


def parse_points(points: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse points string "x,y,label;..." into tensors expected by SAM-I2V.
    Label: 1 foreground, 0 background.
    """
    coords = []
    labels = []
    for token in points.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 3:
            raise ValueError(f"Point '{token}' is not in x,y,label format")
        x, y, lab = parts
        coords.append([float(x), float(y)])
        labels.append(int(lab))

    if not coords:
        raise ValueError("No valid points parsed")

    points_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    labels_t = torch.tensor(labels, dtype=torch.int32).unsqueeze(0)
    return points_t, labels_t


def save_mask(mask: np.ndarray, out_dir: str, frame_name: str, target_shape: Tuple[int, int]) -> None:
    """Save a binary mask as uint8 PNG with the given basename."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(frame_name)[0] + ".png")
    mask_np = np.asarray(mask)
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 1 and mask_np.size == target_shape[0] * target_shape[1]:
        mask_np = mask_np.reshape(target_shape)
    elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    if mask_np.ndim != 2:
        raise ValueError(f"Mask for {frame_name} has unexpected shape {mask_np.shape}")
    mask_np = (mask_np > 0).astype(np.uint8)
    Image.fromarray(mask_np).save(out_path)


def suggest_box_from_frame(frame_path: str, min_area_frac: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Heuristic: build several foreground cues (saliency if available, gradients with
    multiple thresholds, deviation from median) and pick the largest blob. Falls back
    to full image if nothing reasonable is found.
    """
    img = cv2.imread(frame_path)
    if img is None:
        raise RuntimeError(f"Failed to read frame for auto-box: {frame_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    candidate_masks = []

    # Saliency if available (OpenCV contrib)
    saliency_mask = None
    try:
        if hasattr(cv2, "saliency"):
            detector = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, sal = detector.computeSaliency(img)
            if ok:
                sal = (sal / (sal.max() + 1e-8)).astype(np.float32)
                _, sal_bin = cv2.threshold((sal * 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
                saliency_mask = (sal_bin > 0).astype(np.uint8)
                candidate_masks.append(saliency_mask)
    except Exception:
        pass

    # Gradient-based masks with multiple quantiles
    lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F))
    lap = lap / (lap.max() + 1e-8)
    for q in (0.9, 0.8, 0.7, 0.6, 0.5):
        thresh = np.quantile(lap, q)
        candidate_masks.append((lap > thresh).astype(np.uint8))

    # Deviation from median intensity
    med = np.median(blur)
    dev = np.abs(blur - med)
    dev = dev / (dev.max() + 1e-8)
    candidate_masks.append((dev > 0.5).astype(np.uint8))

    kernel = np.ones((5, 5), np.uint8)
    best = None
    best_area = -1.0
    for m in candidate_masks:
        mask = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = int(np.argmax(areas))
        area = areas[max_idx]
        if area < min_area_frac * h * w:
            continue
        if area > best_area:
            best_area = area
            x, y, bw, bh = cv2.boundingRect(contours[max_idx])
            best = (float(x), float(y), float(x + bw), float(y + bh))

    if best is None:
        return (0.0, 0.0, float(w), float(h))
    return best


def maybe_adjust_box(box: Tuple[float, float, float, float], shape: Tuple[int, int], auto_accept: bool) -> Tuple[float, float, float, float]:
    """If interactive, let user accept/override the auto box; otherwise return as-is."""
    h, w = shape
    print(f"Auto-suggested box: {box} (xmin ymin xmax ymax)")
    if auto_accept or not sys.stdin.isatty():
        return box
    resp = input("Accept box? [Y/n/custom 'x1 y1 x2 y2']: ").strip()
    if resp == "" or resp.lower().startswith("y"):
        return box
    parts = resp.replace(",", " ").split()
    if len(parts) == 4:
        try:
            nums = [float(p) for p in parts]
            nums[0] = max(0.0, min(nums[0], w))
            nums[2] = max(0.0, min(nums[2], w))
            nums[1] = max(0.0, min(nums[1], h))
            nums[3] = max(0.0, min(nums[3], h))
            return tuple(nums)  # type: ignore
        except Exception:
            print("Could not parse custom box; using auto box.")
    return box


def main():
    parser = argparse.ArgumentParser(description="Run SAM-I2V to export object masks for Gaussian Grouping.")
    parser.add_argument("--frames", required=True, help="Directory of input frames (JPEG/PNG) used by COLMAP.")
    parser.add_argument("--output", required=True, help="Directory to write masks (will mirror frame basenames).")
    parser.add_argument("--sam-i2v-repo", required=True, help="Path to SAM-I2V repo (for imports).")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM-I2V checkpoint (sam-i2v_*.pt).")
    parser.add_argument("--config", required=True, help="Path to SAM-I2V config YAML (e.g., i2v-infer.yaml).")
    parser.add_argument("--device", default="cuda", help="Device for inference (cuda or cpu).")
    parser.add_argument("--prompt-frame", type=int, default=0, help="Frame index (sorted order) to place the prompt.")
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Bounding box prompt in input pixel coords.",
    )
    parser.add_argument(
        "--points",
        type=str,
        help='Semicolon-separated points "x,y,label;..."; label 1=fg, 0=bg. Used if no box is given.',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Mask logits threshold; > threshold => foreground (default 0.0 matches SAM-I2V tests).",
    )
    parser.add_argument(
        "--auto-box",
        action="store_true",
        help="Auto-select a box on the prompt frame if no box/points are provided.",
    )
    parser.add_argument(
        "--auto-box-frame",
        type=int,
        default=0,
        help="Frame index (sorted order) used for auto box suggestion (default 0).",
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Skip interactive confirmation of the auto box (useful for batch runs).",
    )
    args = parser.parse_args()

    # Add SAM-I2V to import path
    sys.path.insert(0, os.path.abspath(args.sam_i2v_repo))

    try:
        from i2v.build_i2v import build_i2v_video_predictor
    except ModuleNotFoundError as exc:
        # hydra-core is the expected dependency; "hydra" is a different/legacy package.
        raise RuntimeError(
            f"Failed to import SAM-I2V from {args.sam_i2v_repo}. "
            "Install SAM-I2V in editable mode (`pip install -e SAM-I2V`) or "
            "install hydra-core directly (`pip install hydra-core>=1.3.2`)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to import SAM-I2V from {args.sam_i2v_repo}") from exc

    if args.box is None and not args.points and not args.auto_box:
        raise ValueError("Provide either --box, --points, or enable --auto-box.")

    frame_names = [
        f
        for f in os.listdir(args.frames)
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
    ]
    if not frame_names:
        raise RuntimeError(f"No frames found in {args.frames}")
    frame_names = sort_frame_names(frame_names)
    first_frame_path = os.path.join(args.frames, frame_names[0])
    with Image.open(first_frame_path) as img0:
        target_shape = (img0.height, img0.width)

    if args.prompt_frame < 0 or args.prompt_frame >= len(frame_names):
        raise ValueError(f"--prompt-frame {args.prompt_frame} out of range (0-{len(frame_names)-1})")
    if args.auto_box_frame < 0 or args.auto_box_frame >= len(frame_names):
        raise ValueError(f"--auto-box-frame {args.auto_box_frame} out of range (0-{len(frame_names)-1})")

    device = torch.device(args.device)
    # Hydra expects config_name relative to the SAM-I2V/i2v directory (no ".yaml").
    config_abs = os.path.abspath(args.config)
    i2v_root = os.path.abspath(os.path.join(args.sam_i2v_repo, "i2v"))
    if config_abs.startswith(i2v_root):
        config_name = os.path.splitext(os.path.relpath(config_abs, i2v_root))[0]
    else:
        # fallback: use the stem
        config_name = os.path.splitext(os.path.basename(config_abs))[0]
    config_name = config_name.replace(os.sep, "/")

    predictor = build_i2v_video_predictor(
        config_name,
        ckpt_path=args.checkpoint,
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )

    # Load video frames; SAM-I2V internally resizes/normalizes.
    state = predictor.init_state(
        video_path=args.frames,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )

    # Prepare prompts
    points_t = None
    labels_t = None
    box = None
    if args.box is not None:
        box = torch.tensor(args.box, dtype=torch.float32)
    elif args.points:
        points_t, labels_t = parse_points(args.points)
    elif args.auto_box:
        frame_for_auto = os.path.join(args.frames, frame_names[args.auto_box_frame])
        suggested_box = suggest_box_from_frame(frame_for_auto)
        adjusted_box = maybe_adjust_box(suggested_box, target_shape, args.auto_accept)
        box = torch.tensor(adjusted_box, dtype=torch.float32)
    else:
        raise ValueError("No prompt provided")

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        # Seed masks with the prompt frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=args.prompt_frame,
            obj_id=0,
            points=points_t,
            labels=labels_t,
            box=box,
        )
        # Save the prompt frame output
        for i, obj_id in enumerate(object_ids):
            if obj_id != 0:
                continue  # only keep the first object
            mask = (masks[i] > args.threshold).cpu().numpy().astype(np.uint8)
            mask[mask > 0] = 1
            save_mask(mask, args.output, frame_names[frame_idx], target_shape)

        # Propagate to all frames
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for i, obj_id in enumerate(object_ids):
                if obj_id != 0:
                    continue
                mask = (masks[i] > args.threshold).cpu().numpy().astype(np.uint8)
                mask[mask > 0] = 1
                save_mask(mask, args.output, frame_names[frame_idx], target_shape)

    print(f"Saved masks to {args.output} for {len(os.listdir(args.output))} frames.")


if __name__ == "__main__":
    main()
