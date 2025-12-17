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
        
        return {"image": image_tensor, "target": image_tensor, "image_path": str(image_path)}

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
    
    # get_peft_model은 일반 PyTorch 모델도 지원하지만 타입 힌트는 PreTrainedModel만 명시
    model = get_peft_model(model, lora_config)  # type: ignore
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

def train_epoch(model, dataloader, optimizer, accelerator, device, loss_type="l1", gradient_accumulation_steps=4):
    """한 에폭 학습
    
    Args:
        gradient_accumulation_steps: Gradient accumulation 스텝 수 (효과적인 배치 크기 = batch_size * gradient_accumulation_steps)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # 텐서를 PIL Image로 변환 (TripoSR은 PIL Image를 받음)
        image_tensors = batch["image"]  # [B, C, H, W]
        pil_images = []
        for img_tensor in image_tensors:
            # 텐서를 PIL Image로 변환
            # 텐서는 [C, H, W] 형태이고, 값 범위가 [-1, 1]로 정규화되어 있음
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            # [-1, 1] 범위를 [0, 255]로 변환
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)
        
        # PEFT 래퍼를 통해 실제 모델에 접근
        # base_model을 통해 TripoSR의 forward 직접 호출
        base_model = model.base_model if hasattr(model, 'base_model') else model
        
        # TripoSR의 forward로 scene_codes 생성
        scene_codes = base_model(pil_images, device=str(device))
        
        # scene_codes로부터 렌더링된 이미지 생성 (self-supervised 학습)
        # 메모리 절약을 위해 렌더링 해상도 줄임 (256 -> 128)
        rendered_images = base_model.render(
            scene_codes,
            n_views=1,
            elevation_deg=0.0,
            camera_distance=1.9,
            fovy_deg=40.0,
            height=128,  # 메모리 절약: 256 -> 128
            width=128,   # 메모리 절약: 256 -> 128
            return_type="pt"
        )
        
        # 렌더링된 이미지는 리스트 형태이므로 첫 번째 뷰 사용
        if isinstance(rendered_images[0], list):
            outputs = rendered_images[0][0]  # 첫 번째 배치, 첫 번째 뷰
        else:
            outputs = rendered_images[0]
        
        # outputs shape 확인 및 변환
        # renderer가 반환하는 형태: [H, W, C] 또는 [C, H, W] 또는 [H, W] 등
        # targets shape: [B, C, H, W] = [1, 3, 128, 128]
        
        # outputs를 [B, C, H, W] 형태로 변환
        if len(outputs.shape) == 2:  # [H, W] -> [1, 1, H, W]
            outputs = outputs.unsqueeze(0).unsqueeze(0)
        elif len(outputs.shape) == 3:
            # [H, W, C] 또는 [C, H, W] 확인
            if outputs.shape[0] == outputs.shape[1]:  # [H, W, C] 형태일 가능성
                outputs = outputs.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            # [C, H, W] -> [1, C, H, W]
            outputs = outputs.unsqueeze(0)
        elif len(outputs.shape) == 4:
            # 이미 [B, C, H, W] 형태
            pass
        else:
            raise ValueError(f"Unexpected outputs shape: {outputs.shape}")
        
        # 타겟 이미지 준비 (텐서 형태로 변환)
        targets = batch["image"]  # [B, C, H, W] = [1, 3, 128, 128]
        
        # 출력과 타겟의 shape 맞추기 (채널 수 확인)
        if outputs.shape[1] != targets.shape[1]:
            # 채널 수가 다르면 조정 (예: grayscale -> RGB)
            if outputs.shape[1] == 1 and targets.shape[1] == 3:
                outputs = outputs.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
            elif outputs.shape[1] == 3 and targets.shape[1] == 1:
                # RGB -> Grayscale 변환
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
        # Gradient accumulation: loss를 accumulation steps로 나눔
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        accumulated_loss += loss.item()
        
        # Gradient accumulation: 지정된 스텝마다 optimizer 업데이트
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += accumulated_loss * gradient_accumulation_steps  # 원래 스케일로 복원
            num_batches += 1
            accumulated_loss = 0.0
            
            if num_batches % 10 == 0:
                accelerator.print(f"Batch {num_batches}, Loss: {total_loss / num_batches:.6f}")
        
        # 메모리 정리
        del scene_codes, rendered_images, outputs, targets, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 마지막 남은 gradient가 있으면 업데이트
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
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    accelerator.print("Loading TripodSR model...")
    model = load_tripodsr_model(device)
    
    accelerator.print("Applying LoRA to model...")
    model = apply_lora_to_model(model, r=4, alpha=32)
    
    data_dir = "data/my_product_dataset"
    # 메모리 절약을 위해 이미지 크기 줄임 (256 -> 128)
    dataset = ImageDataset(data_dir, image_size=128)
    
    if len(dataset) == 0:
        accelerator.print(f"No .jpg images found in {data_dir}")
        return
    
    accelerator.print(f"Found {len(dataset)} images")
    
    # 메모리 절약: 배치 크기 1, gradient accumulation으로 효과적인 배치 크기 유지
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    num_epochs = 10
    accelerator.print(f"Starting training for {num_epochs} epochs...")
    
    # Gradient accumulation으로 효과적인 배치 크기 4 유지 (실제 배치 크기 1 * 4)
    gradient_accumulation_steps = 4
    
    for epoch in range(num_epochs):
        accelerator.print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, accelerator, device, gradient_accumulation_steps=gradient_accumulation_steps)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        # 에폭마다 메모리 정리
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accelerator.print("\nSaving LoRA weights...")
    unwrapped_model = accelerator.unwrap_model(model)
    save_lora_weights(unwrapped_model, "/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors")
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()