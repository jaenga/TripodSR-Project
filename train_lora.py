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
        
        self.image_paths.extend(self.image_dir.glob("*.jpg"))
        self.image_paths.extend(self.image_dir.glob("*.JPG"))
        self.image_paths.extend(self.image_dir.glob("*.jpeg"))
        self.image_paths.extend(self.image_dir.glob("*.JPEG"))
        self.image_paths.extend(self.image_dir.glob("*.png"))
        self.image_paths.extend(self.image_dir.glob("*.PNG"))
        self.image_paths = sorted(self.image_paths)
        
        # TripoSR은 PIL Image를 직접 받으므로 transform은 사용하지 않음
        # 하지만 학습을 위해 텐서로 변환하는 transform 유지
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
        
        return {"image": image_tensor, "target": image_tensor, "pil_image": image}

def apply_lora_to_model(model: nn.Module, r: int = 4, alpha: int = 32):
    """Apply LoRA to all layers with 'attn' in their name."""
    
    target_modules: List[str] = []
    for name, module in model.named_modules():
        if "attn" in name.lower():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                target_modules.append(name)
    
    if not target_modules:
        print("Warning: No standard layers found. Using module paths.")
        target_modules = []
        for name, module in model.named_modules():
            if "attn" in name.lower() and len(list(module.children())) == 0:
                target_modules.append(name)
        
        if not target_modules:
            raise ValueError("No modules with 'attn' in name found for LoRA application")
    
    print(f"Applying LoRA to modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def load_tripodsr_model(device: torch.device):
    """TripodSR 베이스 모델을 로드합니다.
    
    Args:
        device: 사용할 디바이스
    
    Returns:
        로드된 TripoSR 모델
    """
    from triposr_backbone import load_tripodsr_model as load_model
    
    model, _ = load_model(device=str(device))
    return model

def compute_loss(model_output, target, loss_type="l1"):
    if loss_type == "l1":
        return F.l1_loss(model_output, target)
    elif loss_type == "mse":
        return F.mse_loss(model_output, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_epoch(model, dataloader, optimizer, accelerator, device, loss_type="l1"):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # TripoSR은 PIL Image를 받으므로 PIL Image 사용
        pil_images = batch["pil_image"]
        
        optimizer.zero_grad()
        
        # TripoSR의 forward로 scene_codes 생성
        scene_codes = model(pil_images, device=str(device))
        
        # scene_codes로부터 렌더링된 이미지 생성 (self-supervised 학습)
        # 여러 뷰에서 렌더링하여 원본 이미지와 비교
        rendered_images = model.render(
            scene_codes,
            n_views=1,
            elevation_deg=0.0,
            camera_distance=1.9,
            fovy_deg=40.0,
            height=256,
            width=256,
            return_type="pt"
        )
        
        # 렌더링된 이미지는 리스트 형태이므로 첫 번째 뷰 사용
        if isinstance(rendered_images[0], list):
            outputs = rendered_images[0][0]  # 첫 번째 배치, 첫 번째 뷰
        else:
            outputs = rendered_images[0]
        
        # 타겟 이미지 준비 (텐서 형태로 변환)
        targets = batch["image"]
        
        # 출력과 타겟의 shape 맞추기
        if outputs.shape != targets.shape:
            outputs = F.interpolate(
                outputs.unsqueeze(0) if len(outputs.shape) == 3 else outputs,
                size=targets.shape[2:] if len(targets.shape) == 4 else targets.shape[1:],
                mode="bilinear",
                align_corners=False
            )
            if len(outputs.shape) == 4 and outputs.shape[0] == 1:
                outputs = outputs.squeeze(0)

        loss = compute_loss(outputs, targets, loss_type)
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            accelerator.print(f"Batch {num_batches}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def save_lora_weights(model, output_path):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_state_dict[name] = param.data
    save_file(lora_state_dict, output_path)
    print(f"LoRA weights saved to {output_path}")

def main():
    create_directories()
    accelerator = Accelerator(mixed_precision="fp16")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    accelerator.print("Loading TripodSR model...")
    model = load_tripodsr_model(device)
    
    accelerator.print("Applying LoRA to model...")
    model = apply_lora_to_model(model, r=4, alpha=32)
    
    data_dir = "data/my_product_dataset"
    dataset = ImageDataset(data_dir, image_size=256)
    
    if len(dataset) == 0:
        accelerator.print(f"No .jpg images found in {data_dir}")
        return
    
    accelerator.print(f"Found {len(dataset)} images")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    num_epochs = 10
    accelerator.print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        accelerator.print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, accelerator, device)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
    
    accelerator.print("\nSaving LoRA weights...")
    unwrapped_model = accelerator.unwrap_model(model)
    save_lora_weights(unwrapped_model, "/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors")
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()