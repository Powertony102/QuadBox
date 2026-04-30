#!/usr/bin/env python3
"""Batch training script for Tanks & Temples scenes."""

import os
import subprocess
import sys
import argparse

SCENES = ["train", "truck"]

def parse_args():
    parser = argparse.ArgumentParser(description="Batch training for Tanks & Temples")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Base path to the Tanks & Temples dataset")
    parser.add_argument("--output_path", type=str, default="output/quadgaussian_tandt",
                       help="Base path for output")
    parser.add_argument("--wandb_project", type=str, default="QuadGaussian",
                       help="Wandb project name")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--scenes", nargs="+", type=str, default=None,
                       help="List of scenes to train (default: all)")
    parser.add_argument("--start_from", type=str, default=None,
                       help="Start training from this scene")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--sparse_adam", action="store_true",
                       help="Use sparse Adam optimizer")
    return parser.parse_args()


def run_training(scene_name, args):
    print(f"\n{'='*60}")
    print(f"Training scene: {scene_name}")
    print(f"{'='*60}")

    data_path = os.path.join(args.data_path, scene_name)
    output_path = os.path.join(args.output_path, scene_name)

    if not os.path.exists(data_path):
        print(f"Error: data path does not exist: {data_path}")
        return False

    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "python", "train.py",
        "-s", data_path,
        "-i", "images",
        "-m", output_path,
        "--eval",
        "--disable_viewer",
        "--quiet",
    ]
    if args.sparse_adam:
        cmd.append("--sparse_adam")

    if not args.disable_wandb:
        cmd.extend([
            "--use_wandb",
            "--wandb_project", args.wandb_project,
            "--wandb_name", scene_name
        ])

    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("Dry run mode: skipping execution")
        return True

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"Scene {scene_name} training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Scene {scene_name} training failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\nTraining interrupted for scene {scene_name}")
        return False


def main():
    args = parse_args()
    scenes_to_train = args.scenes if args.scenes else SCENES

    if args.start_from:
        try:
            start_index = scenes_to_train.index(args.start_from)
            scenes_to_train = scenes_to_train[start_index:]
            print(f"Starting from scene '{args.start_from}'")
        except ValueError:
            print(f"Error: start scene '{args.start_from}' not found")
            sys.exit(1)

    print(f"Scenes to train: {', '.join(scenes_to_train)}")
    print(f"Data base path: {args.data_path}")
    print(f"Output base path: {args.output_path}")

    if not os.path.exists(args.data_path):
        print(f"Error: data base path does not exist: {args.data_path}")
        sys.exit(1)

    if not os.path.exists("train.py"):
        print("Error: train.py not found. Run this script from the project root.")
        sys.exit(1)

    total_scenes = len(scenes_to_train)
    successful_scenes = 0
    failed_scenes = []

    for i, scene in enumerate(scenes_to_train, 1):
        print(f"\nProgress: {i}/{total_scenes}")
        success = run_training(scene, args)
        if success:
            successful_scenes += 1
        else:
            failed_scenes.append(scene)

    print(f"\n{'='*60}")
    print("Training summary:")
    print(f"{'='*60}")
    print(f"Total scenes: {total_scenes}")
    print(f"Successful: {successful_scenes}")
    print(f"Failed: {len(failed_scenes)}")
    if failed_scenes:
        print(f"Failed scenes: {', '.join(failed_scenes)}")
    else:
        print("All scenes trained successfully!")


if __name__ == "__main__":
    main()
