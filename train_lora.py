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
import numpy as np

# 필요한 디렉토리 만들기
def create_directories():
    os.makedirs("data/my_product_dataset", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/tripodsr/checkpoints", exist_ok=True)

# 이미지 데이터셋 클래스
class ImageDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.image_paths = []
        
        # PNG 파일만 가져오기
        self.image_paths.extend(self.image_dir.glob("*.png"))
        self.image_paths.extend(self.image_dir.glob("*.PNG"))
        self.image_paths = sorted(self.image_paths)
        
        # 이미지 전처리
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
        
        return {"image": image_tensor, "target": image_tensor, "image_path": str(image_path)}

# LoRA 적용하기
def apply_lora_to_model(model: nn.Module, r: int = 4, alpha: int = 32):
    # attention 레이어 찾기
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
    
    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)  # type: ignore
    model.print_trainable_parameters()
    
    return model

# TripoSR 모델 불러오기
def load_tripodsr_model(device: torch.device):
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

# 한 에폭 학습하기
def train_epoch(model, dataloader, optimizer, accelerator, device, loss_type="l1", gradient_accumulation_steps=4):
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # 텐서를 PIL 이미지로 변환
        image_tensors = batch["image"]
        pil_images = []
        for img_tensor in image_tensors:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)
        
        # 모델 forward
        base_model = model.base_model if hasattr(model, 'base_model') else model
        scene_codes = base_model(pil_images, device=str(device))
        
        # 렌더링해서 이미지 생성
        from tsr.utils import get_spherical_cameras  # type: ignore
        
        n_views = 1
        elevation_deg = 0.0
        camera_distance = 1.9
        fovy_deg = 40.0
        height = 128
        width = 128
        
        # 카메라 설정
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o = rays_o.to(scene_codes.device)
        rays_d = rays_d.to(scene_codes.device)
        
        # 렌더링
        scene_code = scene_codes[0:1]
        rendered_image = base_model.renderer(
            base_model.decoder, 
            scene_code, 
            rays_o[0:1],
            rays_d[0:1]
        )
        
        outputs = rendered_image[0]
        
        # shape 맞추기
        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(0).unsqueeze(0)
        elif len(outputs.shape) == 3:
            if outputs.shape[0] == outputs.shape[1]:
                outputs = outputs.permute(2, 0, 1)
            outputs = outputs.unsqueeze(0)
        elif len(outputs.shape) == 4:
            pass
        else:
            raise ValueError(f"Unexpected outputs shape: {outputs.shape}")
        
        targets = batch["image"]
        
        # 채널 수 맞추기
        if outputs.shape[1] != targets.shape[1]:
            if outputs.shape[1] == 1 and targets.shape[1] == 3:
                outputs = outputs.repeat(1, 3, 1, 1)
            elif outputs.shape[1] == 3 and targets.shape[1] == 1:
                outputs = outputs.mean(dim=1, keepdim=True)
        
        # 해상도 맞추기
        if outputs.shape[2:] != targets.shape[2:]:
            outputs = F.interpolate(
                outputs,
                size=targets.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        loss = compute_loss(outputs, targets, loss_type)
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        accumulated_loss += loss.item()
        
        # gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += accumulated_loss * gradient_accumulation_steps
            num_batches += 1
            accumulated_loss = 0.0
            
            if num_batches % 10 == 0:
                accelerator.print(f"Batch {num_batches}, Loss: {total_loss / num_batches:.6f}")
        
        # 메모리 정리
        del scene_codes, rendered_image, outputs, targets, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 남은 gradient 처리
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss * gradient_accumulation_steps
        num_batches += 1
    
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
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    accelerator.print("Loading TripodSR model...")
    model = load_tripodsr_model(device)
    
    accelerator.print("Applying LoRA to model...")
    model = apply_lora_to_model(model, r=4, alpha=32)
    
    # 데이터셋 불러오기
    DATASET_DIR = "data/my_product_dataset/no_background"
    dataset = ImageDataset(DATASET_DIR, image_size=128)
    
    if len(dataset) == 0:
        accelerator.print(f"No PNG images found in {DATASET_DIR}")
        return
    
    accelerator.print(f"Found {len(dataset)} images")
    
    # 데이터로더 설정
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    num_epochs = 10
    accelerator.print(f"Starting training for {num_epochs} epochs...")
    
    gradient_accumulation_steps = 4
    
    # 학습 시작
    for epoch in range(num_epochs):
        accelerator.print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, accelerator, device, gradient_accumulation_steps=gradient_accumulation_steps)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 가중치 저장
    accelerator.print("\nSaving LoRA weights...")
    unwrapped_model = accelerator.unwrap_model(model)
    save_lora_weights(unwrapped_model, "/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors")
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()