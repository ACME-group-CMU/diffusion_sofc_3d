import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything

# Import the original Diffusion class from your existing diffusion.py file
# Ensure diffusion.py is in the same directory or accessible in PYTHONPATH
from diffusion import Diffusion

import warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


# New class inheriting from Diffusion, containing the predict_step
class DiffusionInference(Diffusion):
    def __init__(self, *args, **kwargs):
        # Call the parent's __init__ to ensure all hparams and model components are set up
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        PyTorch Lightning predict_step.
        Generates a single batch of samples based on parameters from the dataloader.
        'batch' is expected to be a dictionary of parameters for self.generate().
        """
        params = batch
        # If DataLoader batch_size = 1, params is a single dict.
        # DataLoader might wrap it in a list if default collate_fn is used.
        if isinstance(params, list) and len(params) == 1:
            params = params[0]

        inf_timesteps = params['inf_timesteps']
        sample_shape = params['sample_shape'] # Expected: (1, C, H, W, D) for one sample
        condition = params['condition']       # Numpy array or None
        w_guidance = params['w']
        use_ema = params['use_ema_model']

        # self.unet and self.ema are inherited from the Diffusion base class
        active_model = self.unet
        if use_ema and self.ema:
            self.ema.apply_shadow()
            active_model = self.ema.model
        
        # self.generate is inherited from the Diffusion base class
        generated_sample_np = self.generate(
            inf_timesteps=inf_timesteps,
            sample_shape=sample_shape,
            condition=condition,
            model=active_model,
            w=w_guidance
        )
        # self.generate() returns a squeezed numpy array: (H,W,D) if C=1, or (C,H,W,D) if C>1.
        # We need to reshape it to a consistent (1, C, H, W, D) for concatenation.
        
        num_channels_expected = sample_shape[1]
        spatial_dims_expected_len = len(sample_shape[2:]) # Should be 3 for H,W,D

        # If channel dim was squeezed (e.g. C=1)
        if generated_sample_np.ndim == spatial_dims_expected_len:
            generated_sample_np = generated_sample_np[np.newaxis, ...] # Add channel dim: (1, H,W,D)
        
        # Add batch dim (which is 1 for this single sample generation)
        generated_sample_np = generated_sample_np[np.newaxis, ...] # Add sample dim: (1, C, H,W,D)

        if use_ema and self.ema:
            self.ema.restore()

        return generated_sample_np # Shape: (1, C, H, W, D)


class InferenceDataset(Dataset):
    """A simple dataset to produce items for the Pytorch Lightning predict loop."""
    def __init__(self, num_total_samples, generation_params_per_sample):
        self.num_total_samples = num_total_samples
        self.generation_params = generation_params_per_sample

    def __len__(self):
        return self.num_total_samples

    def __getitem__(self, idx):
        return self.generation_params


def main(args):
    seed_everything(args.seed, workers=True)

    # 1. Load Model from checkpoint using the new DiffusionInference class
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    # Load using DiffusionInference.load_from_checkpoint.
    # This will instantiate DiffusionInference, call super().__init__ (which is Diffusion.__init__)
    # with hparams from the checkpoint, and then load the state_dict.
    model = DiffusionInference.load_from_checkpoint(
        args.checkpoint_path,
        map_location='cpu',
        strict=False # Use strict=False for robustness when loading into a subclass
    )
    model.eval()

    # 2. Prepare parameters for generation
    sample_shape_single = (1, args.channels, args.img_size, args.img_size, args.img_size)

    condition_parsed = None
    if args.condition:
        try:
            condition_parsed = np.array([float(c.strip()) for c in args.condition.split(',')])
            # Ensure condition_dim is available in hparams before checking
            hparams_condition_dim = model.hparams.get('condition_dim', None)
            if hparams_condition_dim is not None and len(condition_parsed) != hparams_condition_dim:
                raise ValueError(f"Condition dimension mismatch. Expected {hparams_condition_dim}, got {len(condition_parsed)}.")
            print(f"Using condition: {condition_parsed}")
        except ValueError as e:
            print(f"Error parsing condition: {e}. Proceeding with unconditional generation or check input.")
            condition_parsed = None
    else:
        print("No condition provided, proceeding with unconditional generation if model supports it or CFG w < 0.")

    inf_timesteps_to_use = args.inf_timesteps if args.inf_timesteps is not None else model.hparams.get('inf_timesteps', 50)
    print(f"Using {inf_timesteps_to_use} inference timesteps.")

    generation_params_per_sample = {
        "inf_timesteps": inf_timesteps_to_use,
        "sample_shape": sample_shape_single,
        "condition": condition_parsed,
        "w": args.w,
        "use_ema_model": args.use_ema
    }

    # 3. Setup Dataset and DataLoader
    inference_dataset = InferenceDataset(args.num_samples, generation_params_per_sample)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False
    )

    # 4. Setup PyTorch Lightning Trainer
    accelerator = "gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = args.gpus if accelerator == "gpu" else 1
    
    strategy = "auto"
    if devices > 1 or args.num_nodes > 1:
        strategy = "ddp"
        print(f"Using DDP strategy with {devices} devices and {args.num_nodes} nodes.")
    else:
        print(f"Using single device strategy on {accelerator}.")

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        logger=False,
        callbacks=[],
        precision=model.hparams.get('precision', 32),
        enable_progress_bar=True
    )

    # 5. Run Prediction
    print(f"Starting generation of {args.num_samples} samples...")
    predictions = trainer.predict(model, dataloaders=inference_dataloader, return_predictions=True)

    # 6. Aggregate Results
    if not predictions or not any(p is not None for p in predictions):
        print("No samples were generated or an issue occurred. Exiting.")
        return

    valid_samples = [p for p in predictions if p is not None and isinstance(p, np.ndarray)]

    if not valid_samples:
        print("No valid (numpy array) samples were generated. Exiting.")
        return
    
    try:
        final_samples_array = np.concatenate(valid_samples, axis=0)
    except ValueError as e:
        print(f"Error concatenating samples: {e}")
        # for i, p_sample in enumerate(valid_samples):
        # print(f"Sample {i} shape: {p_sample.shape if isinstance(p_sample, np.ndarray) else type(p_sample)}")
        return
        
    print(f"Successfully generated {final_samples_array.shape[0]} samples.")
    print(f"Final array shape: {final_samples_array.shape}")

    # 7. Save Output
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Saving samples to {args.output_path}...")
    np.savez_compressed(args.output_path, samples=final_samples_array)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a Diffusion model checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated samples (.npz file).")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of samples to generate.")
    parser.add_argument("--img_size", type=int, required=True, help="Size of one dimension of the (cubic) image (e.g., 64 for 64x64x64).")
    
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels (default: 1).")
    parser.add_argument("--inf_timesteps", type=int, default=None, help="Number of inference timesteps. Defaults to model's hparams.inf_timesteps or 50.")
    parser.add_argument("--condition", type=str, default=None, help="Condition for generation, comma-separated floats (e.g., '0.1,0.5,0.4').")
    parser.add_argument("--w", type=float, default=3.0, help="Classifier-free guidance weight (default: 3.0).")
    parser.add_argument("--use_ema", action='store_true', help="Use EMA weights for generation.")

    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node (0 for CPU, -1 for all available).")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed generation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers per GPU (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    if args.gpus == -1:
        if torch.cuda.is_available():
            args.gpus = torch.cuda.device_count()
        else:
            print("CUDA not available, --gpus -1 specified. Setting to 0 for CPU.")
            args.gpus = 0
    
    if args.gpus == 0:
        print("Running on CPU.")
    elif not torch.cuda.is_available():
        print("CUDA not available, --gpus > 0 specified. Attempting CPU.")
        args.gpus = 0

    main(args)