import os
import argparse
import numpy as np
from typing import Optional, List, Union, Dict, Any
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter
from diffusion import Diffusion
import warnings

warnings.filterwarnings("ignore")

class DiffusionInference(Diffusion):
    """
    Inference-specific Diffusion model that inherits all functionality
    but adds predict_step for multi-GPU generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        """
        Generate samples for the given batch using the new InferenceDataset format.
        Returns dict with generated samples and conditions.
        """
        if not hasattr(self, "inference_config"):
            raise AttributeError(
                "Model is missing 'inference_config'. Set this before calling trainer.predict."
            )

        config = self.inference_config

        # Use existing get_input method to parse batch format
        noise, condition = self.get_input(batch)
        
        if hasattr(self, 'global_rank'):
            rank = self.global_rank
        else:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
        if batch_idx==0:
            seed_everything(42+rank)
        
        
        print(f"=== PREDICT STEP DEBUG ===")
        print(f"GPU Rank: {rank}, Batch idx: {batch_idx}")
        print(f"PyTorch RNG state: {torch.get_rng_state()[0].clone().cpu().detach().numpy()}")
        
        # Check the batch content
        #noise, condition = self.get_input(batch)
        print(f"Batch noise shape: {noise.shape}")
        print(f"First noise value: {noise[0, 0, 0, 0, 0].clone().cpu().detach().numpy()}")
        if condition is not None:
            print(f"Condition shape: {condition.shape}")
        print(f"=========================")

        batch_size = noise.shape[0]  # Get batch size from noise tensor

        # Create sample shape for this batch
        sample_shape = (
            batch_size,
            config["channels"],
            config["img_size"],
            config["img_size"],
            config["img_size"],
        )

        # Process conditions: convert placeholder zeros to None for unconditional
        processed_conditions = None
        if config["is_conditional"] and condition is not None:
            # Check if conditions are actual values or placeholder zeros
            if torch.any(condition != 0):
                processed_conditions = condition.cpu().numpy()

        # Prepare noise for generation (always a tensor now)
        predetermined_noise = noise.to(self.device)

        if not config["use_ema"]:
            self.hparams.validate_with_ema = False

        # Generate samples
        generated_samples = self.generate(
            inf_timesteps=config["inf_timesteps"],
            sample_shape=sample_shape,
            condition=processed_conditions,
            w=config["w_guidance"],
            noise=predetermined_noise,  # Pass predetermined noise
        )

        # Ensure proper shape (add channel dim if squeezed)
        if config["channels"] == 1 and generated_samples.ndim == 4:
            generated_samples = generated_samples[:, np.newaxis, ...]

        return {
            "samples": generated_samples,
            "conditions": processed_conditions,
            "batch_idx": batch_idx,
            "gpu_rank": self.global_rank if hasattr(self, "global_rank") else 0,
        }


class DistributedSampleWriter(BasePredictionWriter):
    """
    Custom prediction writer that handles saving generated samples
    from distributed inference across multiple GPUs.
    """

    def __init__(self, output_path: str, metadata: dict):
        super().__init__(write_interval="epoch")
        self.output_path = output_path
        self.metadata = metadata
        self.temp_dir = os.path.join(os.path.dirname(output_path), "temp_predictions")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """
        Called at the end of prediction epoch. Saves predictions from each GPU
        to temporary files, then combines them on rank 0.
        """
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # Collect all samples and conditions from this GPU
        gpu_samples = []
        gpu_conditions = []

        for pred_batch in predictions:
            if pred_batch and "samples" in pred_batch:
                gpu_samples.append(pred_batch["samples"])
                if pred_batch["conditions"] is not None:
                    gpu_conditions.append(pred_batch["conditions"])

        # Concatenate samples from this GPU
        if gpu_samples:
            all_gpu_samples = np.concatenate(gpu_samples, axis=0)
            all_gpu_conditions = (
                np.concatenate(gpu_conditions, axis=0) if gpu_conditions else None
            )
        else:
            all_gpu_samples = np.array([])
            all_gpu_conditions = None

        # Save this GPU's results to temporary file
        temp_file = os.path.join(self.temp_dir, f"gpu_{trainer.global_rank}.npz")
        save_data = {"samples": all_gpu_samples}
        if all_gpu_conditions is not None:
            save_data["conditions"] = all_gpu_conditions

        np.savez_compressed(temp_file, **save_data)

        # Barrier to ensure all GPUs have saved their files
        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier()

        # Rank 0 combines all files
        if trainer.is_global_zero:
            self._combine_and_save_final_results(trainer)

    @rank_zero_only
    def _combine_and_save_final_results(self, trainer):
        """
        Combine results from all GPUs and save final output file.
        Only runs on rank 0.
        """
        print("Combining results from all GPUs...")

        all_samples = []
        all_conditions = []

        # Load and combine results from all GPU files
        gpu_files = sorted(
            [
                f
                for f in os.listdir(self.temp_dir)
                if f.startswith("gpu_") and f.endswith(".npz")
            ]
        )

        for gpu_file in gpu_files:
            gpu_path = os.path.join(self.temp_dir, gpu_file)
            if os.path.exists(gpu_path):
                gpu_data = np.load(gpu_path)
                if "samples" in gpu_data and gpu_data["samples"].size > 0:
                    all_samples.append(gpu_data["samples"])
                    if "conditions" in gpu_data:
                        all_conditions.append(gpu_data["conditions"])

        if not all_samples:
            raise RuntimeError("No samples were generated across any GPU!")

        # Concatenate all samples
        final_samples = np.concatenate(all_samples, axis=0)
        final_conditions = (
            np.concatenate(all_conditions, axis=0) if all_conditions else None
        )

        # Trim to exact requested number (in case of padding from uneven distribution)
        requested_samples = self.metadata["num_samples_requested"]
        final_samples = final_samples[:requested_samples]
        if final_conditions is not None:
            final_conditions = final_conditions[:requested_samples]

        final_samples = final_samples.squeeze()
        final_samples = (((final_samples + 1) / 2) * 255).astype(np.uint8)
        # Prepare final save data
        save_data = {
            "samples": final_samples,
            "metadata": {
                **self.metadata,
                "num_samples_generated": final_samples.shape[0],
            },
        }

        if final_conditions is not None:
            save_data["conditions"] = final_conditions

        # Save final results
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        np.savez_compressed(self.output_path, **save_data)

        # Print results
        print(f"✅ Generation completed successfully!")
        print(
            f"📊 Generated {final_samples.shape[0]} samples (requested {requested_samples})"
        )
        print(f"📏 Sample shape: {final_samples.shape}")
        if final_conditions is not None:
            print(f"🎯 Conditions shape: {final_conditions.shape}")
        print(f"💾 Results saved to: {self.output_path}")

        # Cleanup temporary files
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary GPU files."""
        try:
            import shutil

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary files from {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not cleanup temp files: {e}")


class InferenceDataset(Dataset):
    """
    Dataset class for diffusion model inference that handles conditions and predetermined noise.
    
    Supports:
    - Conditional and unconditional generation
    - Predetermined noise from file or random noise generation
    - Flexible sample count determination
    - GPU-friendly distribution via DataLoader
    """

    def __init__(
        self,
        condition_file: Optional[str] = None,
        num_samples: Optional[int] = None,
        condition_dim: Optional[int] = None,
        noise_file: Optional[str] = None,
        img_size: int = 96,
        channels: int = 1,
    ):
        """
        Initialize the InferenceDataset.
        
        Args:
            condition_file: Path to .npy file with conditions of shape (N, C), or None for unconditional
            num_samples: Number of samples (used only if condition_file is None)
            condition_dim: Expected condition dimension from model (None for unconditional models)
            noise_file: Path to .npy file with predetermined noise of shape (N, C, H, W, D), or None for random
            img_size: Image size (cubic)
            channels: Number of image channels
        """
        self.condition_file = condition_file
        self.num_samples = num_samples
        self.condition_dim = condition_dim
        self.noise_file = noise_file
        self.img_size = img_size
        self.channels = channels

        # Initialize data containers
        self.conditions = None
        self.noise = None
        self.dataset_size = 0

        # Load and validate data
        self._load_conditions()
        self._load_noise()
        self._validate_compatibility()

    def _load_conditions(self):
        """Load conditions from file or create placeholders."""
        if self.condition_file is not None:
            # Load conditions from file
            if not os.path.exists(self.condition_file):
                raise FileNotFoundError(
                    f"Condition file not found: {self.condition_file}"
                )

            try:
                conditions_array = np.load(self.condition_file)
            except Exception as e:
                raise ValueError(
                    f"Failed to load condition file {self.condition_file}: {e}"
                )

            # Validate shape
            if conditions_array.ndim != 2:
                raise ValueError(
                    f"Condition file must contain 2D array (N, C), got shape {conditions_array.shape}"
                )

            self.dataset_size, file_condition_dim = conditions_array.shape

            # Validate condition dimension against model
            if (
                self.condition_dim is not None
                and file_condition_dim != self.condition_dim
            ):
                raise ValueError(
                    f"Condition dimension mismatch: model expects {self.condition_dim}D, "
                    f"file contains {file_condition_dim}D conditions"
                )

            self.conditions = torch.tensor(conditions_array, dtype=torch.float32)
            print(f"📁 Loaded {self.dataset_size} conditions from {self.condition_file}")

        else:
            # Unconditional generation or model supports conditioning but we want unconditional
            if self.num_samples is None:
                raise ValueError(
                    "num_samples must be provided when condition_file is None"
                )

            self.dataset_size = self.num_samples

            if self.condition_dim is not None:
                # Model supports conditioning but we want unconditional - use zeros as placeholder
                self.conditions = torch.zeros(
                    (self.dataset_size, self.condition_dim), dtype=torch.float32
                )
                print(
                    f"🎲 Created {self.dataset_size} placeholder conditions for conditional model (unconditional generation)"
                )
            else:
                # Pure unconditional model - no conditions needed
                self.conditions = None
                print(f"🎲 Generating {self.dataset_size} unconditional samples")

    def _load_noise(self):
        """Load predetermined noise from file or flag for random generation."""
        if self.noise_file is not None:
            if not os.path.exists(self.noise_file):
                raise FileNotFoundError(f"Noise file not found: {self.noise_file}")

            try:
                noise_array = np.load(self.noise_file)
            except Exception as e:
                raise ValueError(f"Failed to load noise file {self.noise_file}: {e}")

            # Validate noise shape
            expected_shape = (
                self.dataset_size,
                self.channels,
                self.img_size,
                self.img_size,
                self.img_size,
            )
            if noise_array.shape != expected_shape:
                raise ValueError(
                    f"Noise shape mismatch: expected {expected_shape}, got {noise_array.shape}"
                )

            self.noise = torch.tensor(noise_array, dtype=torch.float32)
            print(f"🎯 Loaded predetermined noise from {self.noise_file}")
            print(f"   Noise shape: {self.noise.shape}")

        else:
            # Will generate random noise per sample
            self.noise = None
            print(f"🎲 Will generate random noise per sample")

    def _validate_compatibility(self):
        """Validate that all loaded data is compatible."""
        if self.dataset_size == 0:
            raise ValueError("Dataset size cannot be zero")

        # If both conditions and noise are loaded, ensure they have the same sample count
        if self.conditions is not None and self.noise is not None:
            if self.conditions.shape[0] != self.noise.shape[0]:
                raise ValueError(
                    f"Sample count mismatch: conditions have {self.conditions.shape[0]} samples, "
                    f"noise has {self.noise.shape[0]} samples"
                )

        print(f"✅ Dataset validation passed: {self.dataset_size} samples ready")

    def __len__(self) -> int:
        """Return the dataset size."""
        return self.dataset_size

    def __getitem__(self, idx: int):
        # Get predetermined noise or generate random noise
        if self.noise is not None:
            noise = self.noise[idx]
        else:
            # Generate random noise for this sample
            noise = torch.randn(
                self.channels, self.img_size, self.img_size, self.img_size
            )
        # Get condition
        condition = None
        if self.conditions is not None:
            condition = self.conditions[idx]

        # Return format: (noise, condition) if conditions exist, else just noise
        if condition is not None:
            return (noise, condition)
        else:
            return noise

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information for logging/debugging."""
        info = {
            "dataset_size": self.dataset_size,
            "condition_file": self.condition_file,
            "noise_file": self.noise_file,
            "condition_dim": self.condition_dim,
            "img_size": self.img_size,
            "channels": self.channels,
            "has_conditions": self.conditions is not None,
            "has_predetermined_noise": self.noise is not None,
        }

        if self.conditions is not None:
            info["condition_shape"] = tuple(self.conditions.shape)
            info["condition_dtype"] = str(self.conditions.dtype)

        if self.noise is not None:
            info["noise_shape"] = tuple(self.noise.shape)
            info["noise_dtype"] = str(self.noise.dtype)

        return info


def create_inference_dataloader(
    condition_file: Optional[str] = None,
    num_samples: Optional[int] = None,
    condition_dim: Optional[int] = None,
    noise_file: Optional[str] = None,
    img_size: int = 96,
    channels: int = 1,
    batch_size_per_gpu: int = 4,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, int, Optional[np.ndarray]]:
    """
    Create a DataLoader for inference using the InferenceDataset.
    
    Args:
        condition_file: Path to .npy file with conditions
        num_samples: Number of samples (used only if condition_file is None)
        condition_dim: Expected condition dimension from model
        noise_file: Path to .npy file with predetermined noise
        img_size: Image size (cubic)
        channels: Number of image channels
        batch_size_per_gpu: Samples per batch per GPU
        num_workers: Number of dataloader workers
        
    Returns:
        tuple: (dataloader, actual_num_samples, conditions_array_for_compatibility)
    """
    dataset = InferenceDataset(
        condition_file=condition_file,
        num_samples=num_samples,
        condition_dim=condition_dim,
        noise_file=noise_file,
        img_size=img_size,
        channels=channels,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,  # Keep order for reproducibility
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=False,  # Process all samples
    )

    # For backward compatibility, extract conditions array
    conditions_array = None
    if dataset.conditions is not None:
        conditions_array = dataset.conditions.cpu().numpy()

    return dataloader, len(dataset), conditions_array


def find_checkpoint_by_version(version: str, checkpoint_type: str = "best_val") -> str:
    """
    Find checkpoint file by version number and type.

    Args:
        version: Version number (e.g., "100162")
        checkpoint_type: Type of checkpoint ("best_train", "best_val", "last")

    Returns:
        Path to checkpoint file
    """
    base_dir = f"lightning_logs/version_{version}/checkpoints"

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}")

    if checkpoint_type == "best_train":
        pattern = "best_loss-*.ckpt"
    elif checkpoint_type == "best_val":
        pattern = "best_val_loss-*.ckpt"
    elif checkpoint_type == "last":
        pattern = "last.ckpt"
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

    import glob

    checkpoint_files = glob.glob(os.path.join(base_dir, pattern))

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found matching {pattern} in {base_dir}"
        )

    # Return the first match (or latest if multiple)
    checkpoint_path = sorted(checkpoint_files)[-1]
    print(f"📂 Using checkpoint: {checkpoint_path}")

    return checkpoint_path


@rank_zero_only
def print_generation_info(
    args, num_gpus: int, model_condition_dim: Optional[int], actual_num_samples: int
):
    """Print comprehensive generation setup information."""
    print("=" * 70)
    print("🚀 MULTI-GPU DIFFUSION INFERENCE SETUP")
    print("=" * 70)
    print(f"📊 Total samples: {actual_num_samples}")
    print(f"🔧 Hardware configuration:")
    print(f"   • GPUs: {num_gpus}")
    print(f"   • Nodes: {args.num_nodes}")
    print(f"   • Samples per GPU batch: {args.batch_size_per_gpu}")
    print(f"   • Workers per GPU: {args.num_workers}")
    print(f"🎨 Generation parameters:")
    print(f"   • Image size: {args.img_size}³")
    print(f"   • Channels: {args.channels}")
    print(f"   • Inference steps: {args.inf_timesteps}")
    print(f"   • Guidance weight: {args.w}")
    print(f"   • Using EMA: {args.use_ema}")
    print(f"🎯 Conditioning:")
    if args.condition_file:
        print(f"   • Type: Conditional")
        print(f"   • Condition file: {args.condition_file}")
        print(f"   • Condition dimension: {model_condition_dim}")
    else:
        print(f"   • Type: Unconditional")
        if model_condition_dim:
            print(f"   • Note: Model supports {model_condition_dim}D conditions")
    print(f"💾 Output: {args.output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Diffusion Model Inference with Lightning"
    )

    # Model and checkpoint arguments
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version number for checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="best_val",
        choices=["best_train", "best_val", "last"],
        help="Type of checkpoint to use",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Specific checkpoint path (overrides version/type)",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated samples (.npz file)",
    )

    # Generation arguments
    parser.add_argument(
        "--condition_file",
        type=str,
        default=None,
        help="Path to .npy file with conditions of shape (N, C)",
    )
    parser.add_argument(
        "--noise_file",
        type=str,
        default=None,
        help="Path to .npy file with pre-determined noise",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of samples (used only if condition_file is None)",
    )
    parser.add_argument("--img_size", type=int, default=96, help="Image size (cubic)")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument(
        "--inf_timesteps",
        type=int,
        default=None,
        help="Number of inference timesteps (default: from model)",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=3.0,
        help="Guidance weight for conditional generation",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Use EMA weights for generation"
    )

    # Training infrastructure
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs per node (-1 for all available)",
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=12,
        help="Number of samples per batch per GPU",
    )

    args = parser.parse_args()

    # Handle GPU configuration
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.gpus > 0 and not torch.cuda.is_available():
        print("⚠️  Warning: GPUs requested but CUDA not available. Using CPU.")
        args.gpus = 0

    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    else:
        checkpoint_path = find_checkpoint_by_version(args.version, args.checkpoint_type)

    seed_everything(42, workers=True)  # Set a fixed seed for reproducibility

    # Load model
    print(f"📥 Loading model from: {checkpoint_path}")
    model = DiffusionInference.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False
    )
    model.eval()

    # Get model's condition dimension
    model_condition_dim = model.hparams.get("condition_dim", None)

    # Create condition dataloader and get actual sample count
    dataloader, actual_num_samples, conditions_array = create_inference_dataloader(
        condition_file=args.condition_file,
        num_samples=args.num_samples,
        condition_dim=model_condition_dim,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        noise_file = args.noise_file,
        img_size = args.img_size,
        channels = args.channels,
    )

    # Validate conditional vs unconditional setup
    is_conditional = args.condition_file is not None
    if is_conditional and model_condition_dim is None:
        print(
            "⚠️  Warning: Conditions provided but model is unconditional. Generating unconditionally."
        )
        is_conditional = False

    # Set inference configuration on model
    effective_inf_timesteps = args.inf_timesteps or model.hparams.get(
        "inf_timesteps", 50
    )
    model.inference_config = {
        "inf_timesteps": effective_inf_timesteps,
        "w_guidance": args.w,
        "use_ema": args.use_ema,
        "channels": args.channels,
        "img_size": args.img_size,
        "is_conditional": is_conditional,
    }

    model = torch.compile(model)

    # Calculate total GPUs and print info
    total_gpus = max(1, args.gpus * args.num_nodes)
    print_generation_info(args, total_gpus, model_condition_dim, actual_num_samples)

    # Prepare metadata
    metadata = {
        "num_samples_requested": actual_num_samples,
        "img_size": args.img_size,
        "channels": args.channels,
        "inf_timesteps": effective_inf_timesteps,
        "w_guidance": args.w,
        "use_ema": args.use_ema,
        "condition_dim": model_condition_dim,
        "is_conditional": is_conditional,
        "condition_file": args.condition_file,
        "checkpoint_path": os.path.basename(checkpoint_path),
        "version": args.version,
        "checkpoint_type": args.checkpoint_type,
        "total_gpus": total_gpus,
    }

    # Create prediction writer
    prediction_writer = DistributedSampleWriter(
        output_path=args.output_path, metadata=metadata
    )

    # Setup trainer
    strategy = "auto"
    if total_gpus > 1:
        strategy = "ddp"  # or "ddp_find_unused_parameters_true" if needed

    trainer = Trainer(
        accelerator="auto",
        devices=args.gpus if args.gpus > 0 else 1,
        num_nodes=args.num_nodes,
        strategy=strategy if args.gpus > 0 else "auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        precision="16-mixed",
        callbacks=[prediction_writer],
        deterministic=True,
    )

    print("🎬 Starting generation...")

    # Run inference - results are handled by the prediction writer
    trainer.predict(model, dataloader, return_predictions=False)

    print("✨ Inference pipeline completed!")


if __name__ == "__main__":
    main()
