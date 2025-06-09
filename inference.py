import os
import argparse
import numpy as np
from typing import Optional, List, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter
from diffusion import Diffusion
import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


class DiffusionInference(Diffusion):
    """
    Inference-specific Diffusion model that inherits all functionality
    but adds predict_step for multi-GPU generation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Generate samples for the given batch.
        Returns dict with generated samples and conditions.
        """
        # Unpack condition batch from TensorDataset
        condition_batch_tensor = batch[0]
        
        if not hasattr(self, 'inference_config'):
            raise AttributeError("Model is missing 'inference_config'. Set this before calling trainer.predict.")
        
        config = self.inference_config
        batch_size = condition_batch_tensor.shape[0]
        
        # Create sample shape for this batch
        sample_shape = (
            batch_size,
            config['channels'],
            config['img_size'],
            config['img_size'],
            config['img_size']
        )
        
        # Process conditions: convert placeholder zeros to None for unconditional
        processed_conditions = None
        if config['is_conditional'] and torch.any(condition_batch_tensor != 0):
            processed_conditions = condition_batch_tensor.cpu().numpy()
        
        # Use EMA model if requested and available
        active_model = self.unet
        if config['use_ema'] and self.ema:
            self.ema.apply_shadow()
            active_model = self.ema.model
        
        # Generate samples
        generated_samples = self.generate(
            inf_timesteps=config['inf_timesteps'],
            sample_shape=sample_shape,
            condition=processed_conditions,
            model=active_model,
            w=config['w_guidance']
        )
        
        # Ensure proper shape (add channel dim if squeezed)
        if config['channels'] == 1 and generated_samples.ndim == 4:
            generated_samples = generated_samples[:, np.newaxis, ...]
        
        # Restore original model if using EMA
        if config['use_ema'] and self.ema:
            self.ema.restore()
        
        return {
            'samples': generated_samples,
            'conditions': processed_conditions,
            'batch_idx': batch_idx,
            'gpu_rank': self.global_rank if hasattr(self, 'global_rank') else 0
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
            if pred_batch and 'samples' in pred_batch:
                gpu_samples.append(pred_batch['samples'])
                if pred_batch['conditions'] is not None:
                    gpu_conditions.append(pred_batch['conditions'])
        
        # Concatenate samples from this GPU
        if gpu_samples:
            all_gpu_samples = np.concatenate(gpu_samples, axis=0)
            all_gpu_conditions = np.concatenate(gpu_conditions, axis=0) if gpu_conditions else None
        else:
            all_gpu_samples = np.array([])
            all_gpu_conditions = None
        
        # Save this GPU's results to temporary file
        temp_file = os.path.join(self.temp_dir, f"gpu_{trainer.global_rank}.npz")
        save_data = {'samples': all_gpu_samples}
        if all_gpu_conditions is not None:
            save_data['conditions'] = all_gpu_conditions
        
        np.savez_compressed(temp_file, **save_data)
        
        # Barrier to ensure all GPUs have saved their files
        if hasattr(trainer.strategy, 'barrier'):
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
        gpu_files = sorted([f for f in os.listdir(self.temp_dir) if f.startswith("gpu_") and f.endswith(".npz")])
        
        for gpu_file in gpu_files:
            gpu_path = os.path.join(self.temp_dir, gpu_file)
            if os.path.exists(gpu_path):
                gpu_data = np.load(gpu_path)
                if 'samples' in gpu_data and gpu_data['samples'].size > 0:
                    all_samples.append(gpu_data['samples'])
                    if 'conditions' in gpu_data:
                        all_conditions.append(gpu_data['conditions'])
        
        if not all_samples:
            raise RuntimeError("No samples were generated across any GPU!")
        
        # Concatenate all samples
        final_samples = np.concatenate(all_samples, axis=0)
        final_conditions = np.concatenate(all_conditions, axis=0) if all_conditions else None
        
        # Trim to exact requested number (in case of padding from uneven distribution)
        requested_samples = self.metadata['num_samples_requested']
        final_samples = final_samples[:requested_samples]
        if final_conditions is not None:
            final_conditions = final_conditions[:requested_samples]
        
        final_samples = final_samples.squeeze()
        # Prepare final save data
        save_data = {
            'samples': final_samples,
            'metadata': {**self.metadata, 'num_samples_generated': final_samples.shape[0]}
        }
        
        if final_conditions is not None:
            save_data['conditions'] = final_conditions
        
        # Save final results
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        np.savez_compressed(self.output_path, **save_data)
        
        # Print results
        print(f"✅ Generation completed successfully!")
        print(f"📊 Generated {final_samples.shape[0]} samples (requested {requested_samples})")
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


def create_condition_dataloader(
    conditions: Optional[Union[List[float], List[List[float]]]], 
    num_samples: int, 
    condition_dim: Optional[int],
    batch_size_per_gpu: int,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a dataloader for conditions that distributes samples across GPUs.
    
    Args:
        conditions: Either None, single condition vector, or list of condition vectors
        num_samples: Total number of samples to generate
        condition_dim: Dimension of each condition vector (None for unconditional)
        batch_size_per_gpu: Samples per batch per GPU
        num_workers: Number of dataloader workers
    """
    if conditions is None or condition_dim is None:
        # Unconditional: create placeholder tensors
        condition_tensor = torch.zeros((num_samples, 1), dtype=torch.float32)
    else:
        # Check if multiple condition vectors or single vector
        if isinstance(conditions[0], (list, tuple, np.ndarray)):
            # Multiple conditions (one per sample)
            if len(conditions) != num_samples:
                raise ValueError(f"Expected {num_samples} condition vectors, got {len(conditions)}")
            
            for i, cond in enumerate(conditions):
                if len(cond) != condition_dim:
                    raise ValueError(f"Condition {i} has {len(cond)} values, expected {condition_dim}")
            
            condition_tensor = torch.tensor(conditions, dtype=torch.float32)
        else:
            # Single condition vector for all samples
            if len(conditions) != condition_dim:
                raise ValueError(f"Expected {condition_dim} condition values, got {len(conditions)}")
            
            condition_array = np.array(conditions, dtype=np.float32)
            condition_tensor = torch.from_numpy(np.tile(condition_array, (num_samples, 1)))
    
    dataset = TensorDataset(condition_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,  # Keep order for reproducibility
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=False  # Process all samples
    )
    
    return dataloader


@rank_zero_only
def print_generation_info(args, num_gpus: int, model_condition_dim: Optional[int]):
    """Print comprehensive generation setup information."""
    print("=" * 70)
    print("🚀 MULTI-GPU DIFFUSION INFERENCE SETUP")
    print("=" * 70)
    print(f"📊 Total samples requested: {args.num_samples}")
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
    if args.condition:
        print(f"   • Type: Conditional")
        print(f"   • Condition dimension: {model_condition_dim}")
        print(f"   • Condition values: {args.condition}")
    else:
        print(f"   • Type: Unconditional")
        if model_condition_dim:
            print(f"   • Note: Model supports {model_condition_dim}D conditions")
    print(f"💾 Output: {args.output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Diffusion Model Inference with Lightning")
    
    # Model and data arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save generated samples (.npz file)")
    parser.add_argument("--num_samples", type=int, default=32,
                       help="Total number of samples to generate")
    parser.add_argument("--img_size", type=int, default=64,
                       help="Image size (cubic)")
    parser.add_argument("--channels", type=int, default=1,
                       help="Number of channels")
    
    # Generation arguments
    parser.add_argument("--inf_timesteps", type=int, default=None,
                       help="Number of inference timesteps (default: from model)")
    parser.add_argument("--condition", type=str, default=None,
                       help="Conditions: 'v1,v2,v3' (same for all) or 'v1,v2;v3,v4;...' (per sample)")
    parser.add_argument("--w", type=float, default=3.0,
                       help="Guidance weight for conditional generation")
    parser.add_argument("--use_ema", action="store_true",
                       help="Use EMA weights for generation")
    
    # Training infrastructure
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs per node (-1 for all available)")
    parser.add_argument("--num_nodes", type=int, default=1,
                       help="Number of nodes")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers for data loading")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4,
                       help="Number of samples per batch per GPU")
    
    args = parser.parse_args()
    
    # Handle GPU configuration
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.gpus > 0 and not torch.cuda.is_available():
        print("⚠️  Warning: GPUs requested but CUDA not available. Using CPU.")
        args.gpus = 0
    
    # Parse conditions
    parsed_conditions = None
    parsed_condition_dim = None
    
    if args.condition:
        if ';' in args.condition:
            # Multiple conditions (one per sample)
            condition_sets = [s.strip() for s in args.condition.split(';') if s.strip()]
            parsed_conditions = []
            for i, cond_set in enumerate(condition_sets):
                cond_values = [float(x.strip()) for x in cond_set.split(',')]
                if i == 0:
                    parsed_condition_dim = len(cond_values)
                elif len(cond_values) != parsed_condition_dim:
                    raise ValueError(f"All condition sets must have same dimension")
                parsed_conditions.append(cond_values)
            
            if len(parsed_conditions) != args.num_samples:
                raise ValueError(f"Provided {len(parsed_conditions)} condition sets "
                               f"but requested {args.num_samples} samples")
        else:
            # Single condition for all samples
            parsed_conditions = [float(x.strip()) for x in args.condition.split(',')]
            parsed_condition_dim = len(parsed_conditions)
    
    # Load model
    print(f"📥 Loading model from: {args.checkpoint_path}")
    model = DiffusionInference.load_from_checkpoint(
        args.checkpoint_path,
        map_location='cpu',
        strict=False
    )
    model.eval()
    # Validate conditions against model
    model_condition_dim = model.hparams.get('condition_dim', None)
    
    if parsed_condition_dim is not None:
        if model_condition_dim is None:
            print("⚠️  Warning: Conditions provided but model is unconditional. Ignoring conditions.")
            parsed_conditions = None
            parsed_condition_dim = None
            args.condition = None
        elif parsed_condition_dim != model_condition_dim:
            raise ValueError(f"Condition dimension mismatch: got {parsed_condition_dim}, "
                           f"model expects {model_condition_dim}")
    
    # Set inference configuration on model
    effective_inf_timesteps = args.inf_timesteps or model.hparams.get('inf_timesteps', 50)
    model.inference_config = {
        'inf_timesteps': effective_inf_timesteps,
        'w_guidance': args.w,
        'use_ema': args.use_ema,
        'channels': args.channels,
        'img_size': args.img_size,
        'is_conditional': parsed_conditions is not None,
    }
    
    
    model = torch.compile(model)
    
    # Calculate total GPUs and print info
    total_gpus = max(1, args.gpus * args.num_nodes)
    print_generation_info(args, total_gpus, model_condition_dim)
    
    # Create dataloader
    dataloader = create_condition_dataloader(
        conditions=parsed_conditions,
        num_samples=args.num_samples,
        condition_dim=parsed_condition_dim,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_workers=args.num_workers,
    )
    
    # Prepare metadata
    metadata = {
        'num_samples_requested': args.num_samples,
        'img_size': args.img_size,
        'channels': args.channels,
        'inf_timesteps': effective_inf_timesteps,
        'w_guidance': args.w,
        'use_ema': args.use_ema,
        'condition_dim': parsed_condition_dim,
        'is_conditional': parsed_conditions is not None,
        'checkpoint_path': os.path.basename(args.checkpoint_path),
        'total_gpus': total_gpus,
    }
    
    # Create prediction writer
    prediction_writer = DistributedSampleWriter(
        output_path=args.output_path,
        metadata=metadata
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
        callbacks=[prediction_writer]
    )
    
    print("🎬 Starting generation...")
    
    # Run inference - results are handled by the prediction writer
    trainer.predict(model, dataloader, return_predictions=False)
    
    print("✨ Inference pipeline completed!")


if __name__ == "__main__":
    main()
