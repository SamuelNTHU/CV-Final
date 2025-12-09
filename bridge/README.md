Bridge pipeline: SAM-I2V masks → Gaussian Grouping
==================================================

This folder wires together the two repos in this workspace:

- `SAM-I2V` for temporally consistent video masks (promptable).
- `Gaussian-Grouping` for COLMAP + Gaussian Splatting reconstruction/segmentation.

What it does
------------
- Extract per-frame masks from SAM-I2V and save them in the format `Gaussian-Grouping` expects (`object_mask/<frame>.png`, background id 0, foreground id 1).
- Keep your video frames and COLMAP poses untouched so Gaussian Grouping can reconstruct the object only.
- Provides lightweight scripts instead of pulling the full DEVA/SAM stack.

Prereqs
-------
- Checkpoints: place your SAM-I2V checkpoint under `SAM-I2V/checkpoints/` (e.g., `sam-i2v_32gpu.pt`).
- Python env: install the two repos in editable mode or set `PYTHONPATH` to include them. Torch 2.5.0/cu121 per SAM-I2V README; Gaussian Grouping uses its own deps (COLMAP, CUDA, etc.). The SAM-I2V package depends on **hydra-core** (not the older `hydra` package), so a quick way to get everything is:
  ```bash
  pip install -e SAM-I2V
  pip install -e Gaussian-Grouping   # if you want imports instead of relative calls
  ```
- Frames: a directory of JPEG/PNG frames for the video (same frames you feed to COLMAP), numbered so sorting by embedded digits matches COLMAP order (e.g., `00000.jpg`, `00001.jpg`, ...).

Workflow (CLI)
--------------
1) Extract frames (if starting from video):
   ```bash
   ffmpeg -i input.mp4 -q:v 2 -start_number 0 frames/'%05d'.jpg
   ```

2) Run COLMAP via Gaussian Grouping to get poses/point cloud:
   ```bash
   cd Gaussian-Grouping
   python convert.py -s ../data/<scene>        # expects images under ../data/<scene>/images
   ```
   (Create `../data/<scene>/images` and copy/link your frames there.)

3) Export masks with SAM-I2V:
   ```bash
   cd ../bridge
   python scripts/export_masks_sam_i2v.py \
     --frames ../data/<scene>/images \
     --output ../data/<scene>/object_mask \
     --sam-i2v-repo ../SAM-I2V \
     --checkpoint ../SAM-I2V/checkpoints/sam-i2v_32gpu.pt \
     --config ../SAM-I2V/i2v/configs/i2v-infer.yaml \
     --prompt-frame 0 \
     --box 120 80 380 420    # xmin ymin xmax ymax in input image coords
   ```
   Output masks are 0/1 uint8 PNGs matching frame basenames.

   If you don't want to hand-pick a box, you can auto-suggest one from the prompt frame:
   ```bash
   python scripts/export_masks_sam_i2v.py \
     --frames ../data/<scene>/images \
     --output ../data/<scene>/object_mask \
     --sam-i2v-repo ../SAM-I2V \
     --checkpoint ../SAM-I2V/checkpoints/sam-i2v_32gpu.pt \
     --config ../SAM-I2V/i2v/configs/i2v-infer.yaml \
     --prompt-frame 0 \
     --auto-box            # heuristic box on frame 0, asks for confirmation if in a TTY
     # add --auto-accept to skip the prompt, or --auto-box-frame N to use a different frame
   ```

4) Run Gaussian Grouping training/inference as usual, pointing at the same scene folder (now containing `images/`, `object_mask/`, `sparse/0/`):
   ```bash
   cd ../Gaussian-Grouping
   bash script/train.sh <scene> 1    # or your preferred config
   ```

Notes
-----
- The exporter assumes a single object; it assigns mask id 1. Extend as needed for multi-object.
- If you want to prompt with points instead of a box, use `--points "x,y,label;x,y,label"` (label 1=fg, 0=bg).
- To integrate into a web UI later, wrap the exporter call after collecting a user box/points on a chosen frame, then invoke the Gaussian Grouping run script.
