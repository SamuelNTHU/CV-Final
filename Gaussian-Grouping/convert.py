#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import subprocess
from typing import Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def run_colmap(cmd: str, task: str, fallback_cmd: Optional[str] = None):
    exit_code = os.system(cmd)
    if exit_code != 0 and fallback_cmd:
        logging.warning(f"{task} failed with code {exit_code}, retrying without GPU.")
        exit_code = os.system(fallback_cmd)
    if exit_code != 0:
        logging.error(f"{task} failed with code {exit_code}. Exiting.")
        exit(exit_code)
    return exit_code


def detect_gpu_flag(colmap_cmd: str, subcommand: str, candidates, default: str) -> str:
    help_cmd = f"{colmap_cmd} {subcommand} --help"
    try:
        proc = subprocess.run(help_cmd, shell=True, capture_output=True, text=True, check=False)
        help_text = (proc.stdout or "") + (proc.stderr or "")
        for candidate in candidates:
            if f"--{candidate}" in help_text:
                return candidate
    except Exception as exc:
        logging.warning(f"Could not inspect {subcommand} help: {exc}")
    logging.warning(f"Falling back to default gpu flag '{default}' for {subcommand}.")
    return default

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0
input_path = os.path.join(args.source_path, "input")
feat_gpu_flag = detect_gpu_flag(
    colmap_command,
    "feature_extractor",
    ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"],
    "FeatureExtraction.use_gpu",
)
match_gpu_flag = detect_gpu_flag(
    colmap_command,
    "exhaustive_matcher",
    ["FeatureMatching.use_gpu", "SiftMatching.use_gpu"],
    "FeatureMatching.use_gpu",
)

if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and use_gpu:
    logging.warning("QT_QPA_PLATFORM=offscreen detected, defaulting to CPU SIFT to avoid OpenGL issues.")
    use_gpu = 0

if not os.path.isdir(input_path):
    logging.error(f"Expected input images at {input_path}, but the directory is missing.")
    exit(1)

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    def feature_extractor_cmd(use_gpu_flag: int) -> str:
        return colmap_command + " feature_extractor "\
            "--database_path " + args.source_path + "/distorted/database.db \
            --image_path " + args.source_path + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + args.camera + " \
            --" + feat_gpu_flag + " " + str(use_gpu_flag)

    feat_extracton_cmd = feature_extractor_cmd(use_gpu)
    fallback_feat_cmd = None if use_gpu == 0 else feature_extractor_cmd(0)
    run_colmap(feat_extracton_cmd, "Feature extraction", fallback_cmd=fallback_feat_cmd)

    ## Feature matching
    def exhaustive_matcher_cmd(use_gpu_flag: int) -> str:
        return colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --" + match_gpu_flag + " " + str(use_gpu_flag)

    feat_matching_cmd = exhaustive_matcher_cmd(use_gpu)
    fallback_match_cmd = None if use_gpu == 0 else exhaustive_matcher_cmd(0)
    run_colmap(feat_matching_cmd, "Feature matching", fallback_cmd=fallback_match_cmd)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
