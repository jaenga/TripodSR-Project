# pip install torch torchvision
# pip install peft accelerate
# pip install Pillow safetensors

import os
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from safetensors.torch import save_file

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("data/my_product_dataset", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/tripodsr/checkpoints", exist_ok=True)

class ImageDataset(Dataset):
    """Dataset for loading images for self-supervised reconstruction training."""
    
    def __init__(self, image_dir: str, image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.image_paths = []
        
        # Load all .jpg images
        self.image_paths.extend(self.image_dir.glob("*.jpg"))
        self.image_paths.extend(self.image_dir.glob("*.JPG"))
        self.image_paths = sorted(self.image_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # For self-supervised reconstruction, input and target are the same
        return {"image": image_tensor, "target": image_tensor}

def apply_lora_to_model(model: nn.Module, r: int = 4, alpha: int = 32):
    """Apply LoRA to all layers with 'attn' in their name."""
    
    # Find all module classes that have 'attn' in their module name
    # PEFT requires target_modules to be class names (e.g., "Linear", "Conv2d")
    target_module_classes = set()
    
    for name, module in model.named_modules():
        if "attn" in name.lower():
            module_class = module.__class__.__name__
            # Only include common linear/conv layers that can have LoRA applied
            if module_class in ["Linear", "Conv1d", "Conv2d", "Conv3d"]:
                target_module_classes.add(module_class)
    
    # Convert to list
    target_modules = list(target_module_classes)
    
    if not target_modules:
        # Fallback: if no standard layers found, try to use module paths
        # This is a workaround for custom modules
        print("Warning: No standard layers found. Using module paths.")
        target_modules = []
        for name, module in model.named_modules():
            if "attn" in name.lower() and len(list(module.children())) == 0:
                # Use full module path for leaf modules
                target_modules.append(name)
        
        if not target_modules:
            raise ValueError("No modules with 'attn' in name found for LoRA application")
    
    print(f"Applying LoRA to modules: {target_modules}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def load_tripodsr_model():
    """Load the TripodSR model.
    
    Note: This is a placeholder function. Replace with actual model loading logic.
    For example:
    - model = TripodSR.from_pretrained(...)
    - model = torch.load("path/to/model.pth")
    - model = YourModelClass()
    """
    # Placeholder: Create a simple example model structure with attention layers
    # Replace this with actual TripodSR model loading
    class ExampleTripodSR(nn.Module):
        def __init__(self):
            super().__init__()
            # Attention layers that will have LoRA applied
            self.attn_conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.attn_linear1 = nn.Linear(64, 256)
            self.attn_conv2 = nn.Conv2d(64, 128, 3, padding=1)
            
            # Regular layers (no LoRA)
            self.other_layer = nn.Linear(256, 128)
            self.final_conv = nn.Conv2d(128, 3, 3, padding=1)
        
        def forward(self, x):
            # Simple forward pass
            x = self.attn_conv1(x)
            x = F.relu(x)
            x = self.attn_conv2(x)
            x = F.relu(x)
            x = self.final_conv(x)
            return x
    
    model = ExampleTripodSR()
    
    # TODO: Replace with actual model loading
    # Example:
    # from tripodsr import TripodSR
    # model = TripodSR.from_pretrained("path/to/model")
    # or
    # model = torch.load("path/to/model.pth", map_location="cpu")
    
    return model

def compute_loss(model_output, target, loss_type="l1"):
    """Compute reconstruction loss."""
    if loss_type == "l1":
        return F.l1_loss(model_output, target)
    elif loss_type == "mse":
        return F.mse_loss(model_output, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    loss_type: str = "l1"
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        images = batch["image"]
        targets = batch["target"]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Resize output to match target if needed
        if outputs.shape != targets.shape:
            outputs = F.interpolate(
                outputs, 
                size=targets.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        # Compute loss
        loss = compute_loss(outputs, targets, loss_type)
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            accelerator.print(f"Batch {num_batches}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def save_lora_weights(model: nn.Module, output_path: str):
    """Save only the LoRA adapter weights."""
    # Get LoRA adapter state dict
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_state_dict[name] = param.data
    
    # Save using safetensors
    save_file(lora_state_dict, output_path)
    print(f"LoRA weights saved to {output_path}")

def main():
    # Create necessary directories
    create_directories()
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Load model
    accelerator.print("Loading TripodSR model...")
    model = load_tripodsr_model()
    
    # Apply LoRA
    accelerator.print("Applying LoRA to model...")
    model = apply_lora_to_model(model, r=4, alpha=32)
    
    # Load dataset
    data_dir = "data/my_product_dataset"
    accelerator.print(f"Loading dataset from {data_dir}...")
    dataset = ImageDataset(data_dir, image_size=256)
    
    if len(dataset) == 0:
        accelerator.print(f"No .jpg images found in {data_dir}")
        return
    
    accelerator.print(f"Found {len(dataset)} images")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Prepare model, optimizer, and dataloader with accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    num_epochs = 10
    accelerator.print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        accelerator.print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, accelerator, loss_type="l1")
        accelerator.print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")
    
    # Save LoRA weights
    accelerator.print("\nSaving LoRA weights...")
    output_path = "/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors"
    
    # Unwrap model if needed
    unwrapped_model = accelerator.unwrap_model(model)
    save_lora_weights(unwrapped_model, output_path)
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()
