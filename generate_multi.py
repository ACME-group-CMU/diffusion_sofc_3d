#!/usr/bin/env python3
"""
Multi-checkpoint generation script for a single version.
Simple sequential execution with flexible checkpoint selection.
"""

import os
import sys
import argparse
import glob
import re
import subprocess
import time
from typing import List, Optional, Tuple


class CheckpointSelector:
    """Handles checkpoint discovery and filtering."""

    def __init__(self, version: str, base_path: str = "./results/lightning_logs"):
        self.version = version
        self.checkpoint_dir = os.path.join(
            base_path, f"version_{version}", "checkpoints"
        )

        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

    def find_all_checkpoints(self) -> List[str]:
        """Find all checkpoint files in the version directory."""
        pattern = os.path.join(self.checkpoint_dir, "*.ckpt")
        checkpoints = glob.glob(pattern)
        return sorted(checkpoints)

    def find_by_specific_paths(self, checkpoint_list: List[str]) -> List[str]:
        """Validate and return specific checkpoint paths."""
        validated_checkpoints = []

        for ckpt in checkpoint_list:
            # Handle relative paths
            if not os.path.isabs(ckpt):
                ckpt_path = os.path.join(self.checkpoint_dir, ckpt)
            else:
                ckpt_path = ckpt

            if os.path.exists(ckpt_path):
                validated_checkpoints.append(ckpt_path)
            else:
                print(f"⚠️  Warning: Checkpoint not found: {ckpt}")

        return validated_checkpoints

    def find_by_epoch_range(self, start_epoch: int, end_epoch: int) -> List[str]:
        """Find checkpoints within an epoch range."""
        all_checkpoints = self.find_all_checkpoints()
        filtered_checkpoints = []

        for ckpt in all_checkpoints:
            epoch = self._extract_epoch(ckpt)
            if epoch is not None and start_epoch <= epoch <= end_epoch:
                filtered_checkpoints.append(ckpt)

        return sorted(filtered_checkpoints, key=lambda x: self._extract_epoch(x) or 0)

    def find_by_step_range(self, start_step: int, end_step: int) -> List[str]:
        """Find checkpoints within a step range."""
        all_checkpoints = self.find_all_checkpoints()
        filtered_checkpoints = []

        for ckpt in all_checkpoints:
            step = self._extract_step(ckpt)
            if step is not None and start_step <= step <= end_step:
                filtered_checkpoints.append(ckpt)

        return sorted(filtered_checkpoints, key=lambda x: self._extract_step(x) or 0)

    def find_by_pattern(self, pattern: str) -> List[str]:
        """Find checkpoints matching a pattern."""
        all_checkpoints = self.find_all_checkpoints()
        filtered_checkpoints = []

        for ckpt in all_checkpoints:
            filename = os.path.basename(ckpt)
            if self._matches_pattern(filename, pattern):
                filtered_checkpoints.append(ckpt)

        return filtered_checkpoints

    def find_latest_n(self, n: int) -> List[str]:
        """Find the N most recent checkpoints by modification time."""
        all_checkpoints = self.find_all_checkpoints()

        # Sort by modification time (most recent first)
        checkpoints_with_time = [
            (ckpt, os.path.getmtime(ckpt)) for ckpt in all_checkpoints
        ]
        checkpoints_with_time.sort(key=lambda x: x[1], reverse=True)

        return [ckpt for ckpt, _ in checkpoints_with_time[:n]]

    def find_periodic_only(self) -> List[str]:
        """Find only periodic checkpoints (exclude best_val, best_train, last)."""
        all_checkpoints = self.find_all_checkpoints()
        filtered_checkpoints = []

        for ckpt in all_checkpoints:
            filename = os.path.basename(ckpt)

            # Skip special checkpoints
            if any(
                pattern in filename
                for pattern in ["best_val_loss", "best_loss", "last.ckpt"]
            ):
                continue

            # Only include files with both epoch and step (periodic checkpoints)
            if "epoch" in filename and "step" in filename:
                filtered_checkpoints.append(ckpt)

        return sorted(filtered_checkpoints, key=lambda x: self._extract_epoch(x) or 0)

    def find_best_checkpoints(self, checkpoint_type: str) -> List[str]:
        """Find best checkpoints (val, train, or last)."""
        if checkpoint_type == "best_val":
            pattern = "best_val_loss-*.ckpt"
        elif checkpoint_type == "best_train":
            pattern = "best_loss-*.ckpt"
        elif checkpoint_type == "last":
            return [os.path.join(self.checkpoint_dir, "last.ckpt")]
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        return sorted(checkpoints)

    def _extract_epoch(self, checkpoint_path: str) -> Optional[int]:
        """Extract epoch number from checkpoint filename."""
        filename = os.path.basename(checkpoint_path)
        match = re.search(r"epoch[=\-](\d+)", filename)
        return int(match.group(1)) if match else None

    def _extract_step(self, checkpoint_path: str) -> Optional[int]:
        """Extract step number from checkpoint filename."""
        filename = os.path.basename(checkpoint_path)
        match = re.search(r"step[=\-](\d+)", filename)
        return int(match.group(1)) if match else None

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a glob-style pattern."""
        import fnmatch

        return fnmatch.fnmatch(filename, pattern)


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse range string like '50-100' into (50, 100)."""
    try:
        start, end = map(int, range_str.split("-"))
        return start, end
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}. Use 'start-end' format.")


def build_inference_command(checkpoint_path: str, output_path: str, args) -> List[str]:
    """Build the inference.py command for a specific checkpoint."""
    cmd = ["python3", "inference.py"]

    # Add checkpoint and output
    cmd.extend(["--checkpoint_path", checkpoint_path])
    cmd.extend(["--output_path", output_path])

    # Add required arguments
    cmd.extend(["--version", args.version])
    cmd.extend(["--num_samples", str(args.num_samples)])
    cmd.extend(["--batch_size_per_gpu", str(args.batch_size_per_gpu)])
    cmd.extend(["--img_size", str(args.img_size)])
    cmd.extend(["--channels", str(args.channels)])
    cmd.extend(["--w", str(args.w)])
    cmd.extend(["--gpus", str(args.gpus)])
    cmd.extend(["--num_nodes", str(args.num_nodes)])
    cmd.extend(["--num_workers", str(args.num_workers)])

    # Add optional arguments
    if args.condition_file:
        cmd.extend(["--condition_file", args.condition_file])
    if args.noise_file:
        cmd.extend(["--noise_file", args.noise_file])
    if args.inf_timesteps:
        cmd.extend(["--inf_timesteps", str(args.inf_timesteps)])
    if args.use_ema:
        cmd.append("--use_ema")

    return cmd


def generate_output_filename(
    checkpoint_path: str, version: str, output_dir: str
) -> str:
    """Generate output filename for a checkpoint."""
    ckpt_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
    return os.path.join(output_dir, f"run_version_{version}_{ckpt_name}.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-checkpoint generation for a single diffusion model version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Generate from specific checkpoints
                python generate_multi.py --version 13 --checkpoints epoch-050.ckpt,epoch-075.ckpt
                
                # Generate from epoch range
                python generate_multi.py --version 13 --epoch_range 50-100
                
                # Generate from pattern
                python generate_multi.py --version 13 --pattern "*best*"
                
                # Generate periodic checkpoints only
                python generate_multi.py --version 13 --periodic_only
                """,
    )

    # Required version argument
    parser.add_argument(
        "--version", type=str, required=True, help="Model version number"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="./results/lightning_logs",
        help="Base path to lightning_logs directory (default: ./results/lightning_logs)",
    )

    # Checkpoint selection arguments (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--checkpoints",
        type=str,
        help="Comma-separated list of checkpoint names or paths",
    )
    selection_group.add_argument(
        "--epoch_range", type=str, help="Epoch range (e.g., '50-100')"
    )
    selection_group.add_argument(
        "--step_range", type=str, help="Step range (e.g., '1000-5000')"
    )
    selection_group.add_argument(
        "--pattern", type=str, help="Glob pattern for checkpoint names (e.g., '*best*')"
    )
    selection_group.add_argument(
        "--latest_n", type=int, help="Use the N most recent checkpoints"
    )
    selection_group.add_argument(
        "--periodic_only",
        action="store_true",
        help="Use only periodic checkpoints (exclude best_val, best_train, last)",
    )
    selection_group.add_argument(
        "--best_type",
        choices=["best_val", "best_train", "last"],
        help="Use specific best checkpoint type",
    )

    # Generation parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (default: ./generated_samples/version_X)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=32, help="Number of samples to generate"
    )
    parser.add_argument("--condition_file", type=str, help="Path to condition file")
    parser.add_argument(
        "--noise_file", type=str, help="Path to predetermined noise file"
    )
    parser.add_argument(
        "--batch_size_per_gpu", type=int, default=12, help="Batch size per GPU"
    )
    parser.add_argument("--img_size", type=int, default=96, help="Image size")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument(
        "--inf_timesteps", type=int, default=100, help="Inference timesteps"
    )
    parser.add_argument("--w", type=float, default=0, help="Guidance weight")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    args = parser.parse_args()

    # Set up output directory
    if not args.output_dir:
        args.output_dir = f"./generated_samples/version_{args.version}"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"🚀 MULTI-CHECKPOINT GENERATION - VERSION {args.version}")
    print(f"{'='*70}")

    try:
        # Initialize checkpoint selector
        selector = CheckpointSelector(args.version)

        # Find checkpoints based on selection criteria
        if args.checkpoints:
            checkpoint_list = [ckpt.strip() for ckpt in args.checkpoints.split(",")]
            checkpoints = selector.find_by_specific_paths(checkpoint_list)
        elif args.epoch_range:
            start, end = parse_range(args.epoch_range)
            checkpoints = selector.find_by_epoch_range(start, end)
        elif args.step_range:
            start, end = parse_range(args.step_range)
            checkpoints = selector.find_by_step_range(start, end)
        elif args.pattern:
            checkpoints = selector.find_by_pattern(args.pattern)
        elif args.latest_n:
            checkpoints = selector.find_latest_n(args.latest_n)
        elif args.periodic_only:
            checkpoints = selector.find_periodic_only()
        elif args.best_type:
            checkpoints = selector.find_best_checkpoints(args.best_type)

        if not checkpoints:
            print(f"❌ No checkpoints found for version {args.version}")
            return 1

        print(f"📊 Found {len(checkpoints)} checkpoints:")
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"   {i:2d}. {os.path.basename(ckpt)}")

        print(f"🎯 Samples per checkpoint: {args.num_samples}")
        print(f"💾 Output directory: {args.output_dir}")
        print(f"{'='*70}")

        # Process each checkpoint sequentially
        success_count = 0
        failed_count = 0

        print(f"\n🎬 Starting generation...")

        for ckpt_idx, checkpoint in enumerate(checkpoints, 1):
            ckpt_name = os.path.basename(checkpoint)
            print(
                f"\n📊 Processing checkpoint {ckpt_idx}/{len(checkpoints)}: {ckpt_name}"
            )

            # Generate output path
            output_path = generate_output_filename(
                checkpoint, args.version, args.output_dir
            )

            # Check if output already exists
            if os.path.exists(output_path):
                print(
                    f"⏭️  Output already exists, skipping: {os.path.basename(output_path)}"
                )
                success_count += 1
                continue

            # Build and execute command
            cmd = build_inference_command(checkpoint, output_path, args)
            start_time = time.time()

            try:
                print(f"🔄 Running inference...")
                result = subprocess.run(cmd, check=True)
                execution_time = time.time() - start_time

                print(f"✅ Completed in {execution_time:.1f}s")
                print(f"💾 Output saved: {os.path.basename(output_path)}")
                success_count += 1

            except subprocess.CalledProcessError as e:
                execution_time = time.time() - start_time
                print(f"❌ Failed after {execution_time:.1f}s")
                print(f"Error code: {e.returncode}")
                failed_count += 1

        # Final summary
        print(f"\n{'='*70}")
        print(f"📋 GENERATION SUMMARY - VERSION {args.version}")
        print(f"{'='*70}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"💾 Output directory: {args.output_dir}")

        if success_count > 0:
            print(f"📄 Generated files:")
            for f in sorted(os.listdir(args.output_dir)):
                if f.startswith(f"run_version_{args.version}_") and f.endswith(".npz"):
                    print(f"   - {f}")

        print(f"{'='*70}")

        return 0 if failed_count == 0 else 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
