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
from argparse import ArgumentParser

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")

parser.add_argument("--output_path", default="./eval")
parser.add_argument("--use_wandb", action="store_true", default=True, help="Enable wandb logging")
parser.add_argument("--wandb_project", type=str, default="QuadGaussian", help="Wandb project name")
parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (optional)")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str,
                       help="Path to MipNeRF-360 dataset")
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str,
                       help="Path to Tanks & Temples dataset")
    parser.add_argument("--deepblending", "-db", required=True, type=str,
                       help="Path to Deep Blending dataset")
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "

    if args.use_wandb:
        common_args += " --use_wandb --wandb_project " + args.wandb_project

    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_wandb_name = f"{scene}_outdoor" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"Training scene: {scene} (outdoor)")
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args + wandb_arg)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_wandb_name = f"{scene}_indoor" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"Training scene: {scene} (indoor)")
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args + wandb_arg)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        scene_wandb_name = f"{scene}_tandt" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"Training scene: {scene} (Tanks & Temples)")
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + wandb_arg)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        scene_wandb_name = f"{scene}_db" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"Training scene: {scene} (Deep Blending)")
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + wandb_arg)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

