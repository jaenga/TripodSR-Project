# pip install torch torchvision
# pip install peft accelerate
# pip install Pillow safetensors

import os
import argparse
import json
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from safetensors.torch import save_file
from triposr_backbone import load_tripodsr_model


def create_directories(checkpoint_dir: str = "checkpoints"):
    """필요한 디렉토리가 없으면 생성합니다."""
    os.makedirs("data/my_product_dataset", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


class ImageDataset(Dataset):
    """이미지 데이터셋 (self-supervised 학습용)
    
    같은 원본 이미지에 서로 다른 augmentation을 적용한 두 버전을 반환합니다.
    """
    
    def __init__(self, image_dir: str, image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.image_paths = []
        
        # 다양한 이미지 확장자 지원
        extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        self.image_paths = sorted(self.image_paths)
        self.image_size = image_size
        
        # 기본 전처리: center crop + resize
        self.base_transform = transforms.Compose([
            transforms.Lambda(lambda img: self._center_crop(img)),
            transforms.Resize((image_size, image_size), Image.Resampling.LANCZOS),
        ])
        
        # Augmentation transform 1: RandomResizedCrop, ColorJitter, HorizontalFlip 등
        self.aug_transform_1 = transforms.Compose([
            transforms.Lambda(lambda img: self._center_crop(img)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=Image.Resampling.LANCZOS
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
        ])
        
        # Augmentation transform 2: 다른 랜덤성으로 적용
        self.aug_transform_2 = transforms.Compose([
            transforms.Lambda(lambda img: self._center_crop(img)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=Image.Resampling.LANCZOS
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
        ])
    
    def _center_crop(self, image: Image.Image) -> Image.Image:
        """이미지를 정사각형으로 center crop합니다."""
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        return image.crop((left, top, right, bottom))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # 같은 원본 이미지에 서로 다른 augmentation 적용
        # 각 transform은 독립적인 랜덤성을 가지므로, deterministic 설정이 켜져 있어도
        # augmentation의 랜덤성 때문에 서로 다른 결과가 나옵니다.
        image_1 = self.aug_transform_1(image)
        image_2 = self.aug_transform_2(image)
        
        return {"image1": image_1, "image2": image_2}


def apply_lora_to_model(model: nn.Module, r: int = 4, alpha: int = 32):
    """TripoSR 모델에 LoRA를 적용합니다.
    
    Args:
        model: TripoSR 모델 인스턴스
        r: LoRA rank
        alpha: LoRA alpha
    
    Returns:
        LoRA가 적용된 모델
    """
    # TripoSR 모델의 attention 레이어 찾기
    # backbone 내부의 attention 레이어를 찾아야 함
    target_modules: List[str] = []
    
    # 먼저 Linear 레이어 중 attention 관련 찾기
    for name, module in model.named_modules():
        name_lower = name.lower()
        # attention 관련 키워드가 포함된 Linear 레이어 찾기
        if ("attn" in name_lower or "attention" in name_lower) and isinstance(module, nn.Linear):
            target_modules.append(name)
    
    # 만약 찾지 못했다면, backbone 내부의 q, k, v, o 프로젝션 레이어 찾기
    if not target_modules:
        print("Warning: 'attn' 또는 'attention'이 포함된 레이어를 찾지 못했습니다.")
        print("backbone 내부의 q, k, v, o 프로젝션 레이어를 찾는 중...")
        
        for name, module in model.named_modules():
            name_lower = name.lower()
            # q, k, v, o 프로젝션 레이어 찾기
            if isinstance(module, nn.Linear) and any(
                key in name_lower for key in ["to_q", "to_k", "to_v", "to_out", "q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                target_modules.append(name)
    
    # 여전히 찾지 못했다면, backbone의 모든 Linear 레이어 사용
    if not target_modules:
        print("Warning: attention 레이어를 찾지 못했습니다. backbone의 모든 Linear 레이어에 LoRA를 적용합니다.")
        for name, module in model.named_modules():
            if "backbone" in name.lower() and isinstance(module, nn.Linear):
                target_modules.append(name)
    
    if not target_modules:
        raise ValueError("LoRA를 적용할 레이어를 찾을 수 없습니다.")
    
    print(f"LoRA를 적용할 모듈 수: {len(target_modules)}")
    print(f"처음 10개 모듈: {target_modules[:10]}")
    
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


def compute_consistency_loss(scene_codes_1, scene_codes_2, loss_type="mse"):
    """두 scene_codes 간의 consistency loss를 계산합니다.
    
    Args:
        scene_codes_1: 첫 번째 scene codes
        scene_codes_2: 두 번째 scene codes
        loss_type: loss 타입 ("mse" 또는 "l1")
    
    Returns:
        loss 값
    """
    if loss_type == "mse":
        return F.mse_loss(scene_codes_1, scene_codes_2)
    elif loss_type == "l1":
        return F.l1_loss(scene_codes_1, scene_codes_2)
    elif loss_type == "cosine":
        # Cosine similarity loss (1 - cosine similarity)
        cos_sim = F.cosine_similarity(
            scene_codes_1.flatten(1),
            scene_codes_2.flatten(1),
            dim=1
        ).mean()
        return 1 - cos_sim
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, dataloader, optimizer, accelerator, device: str, loss_type="mse"):
    """한 에폭 학습
    
    같은 원본 이미지에 서로 다른 augmentation을 적용한 두 버전에 대해
    consistency loss를 계산합니다.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images_1 = batch["image1"]  # 첫 번째 augmentation이 적용된 PIL Image 리스트
        images_2 = batch["image2"]  # 두 번째 augmentation이 적용된 PIL Image 리스트
        
        optimizer.zero_grad()
        
        # 서로 다른 augmentation이 적용된 두 이미지를 각각 forward
        scene_codes_1 = model(images_1, device=device)
        scene_codes_2 = model(images_2, device=device)
        
        # Consistency loss 계산: 같은 원본 이미지에서 나온 scene_codes는 유사해야 함
        loss = compute_consistency_loss(scene_codes_1, scene_codes_2, loss_type=loss_type)
        
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            accelerator.print(f"Batch {num_batches}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_lora_weights(model, output_path: str, r: int, alpha: int):
    """LoRA 가중치와 메타데이터를 저장합니다.
    
    Args:
        model: LoRA가 적용된 모델
        output_path: 저장할 파일 경로 (.safetensors)
        r: LoRA rank
        alpha: LoRA alpha
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_state_dict[name] = param.data.cpu()
    
    if not lora_state_dict:
        print("Warning: 저장할 LoRA 가중치가 없습니다.")
        return
    
    # LoRA 가중치 저장
    save_file(lora_state_dict, output_path)
    print(f"LoRA 가중치 저장 완료: {output_path}")
    print(f"저장된 파라미터 수: {len(lora_state_dict)}")
    
    # 메타데이터 JSON 저장 (같은 디렉토리에)
    metadata_path = output_path.replace(".safetensors", "_config.json")
    metadata = {
        "r": r,
        "alpha": alpha,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION"
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"LoRA 메타데이터 저장 완료: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="TripoSR 모델에 LoRA 파인튜닝")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/my_product_dataset",
        help="학습 이미지 디렉토리"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="체크포인트 저장 디렉토리"
    )
    parser.add_argument(
        "--r",
        type=int,
        default=4,
        help="LoRA rank"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="배치 크기 (Colab GPU 메모리 고려)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="학습률"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="입력 이미지 크기"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "l1", "cosine"],
        help="Loss 타입"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="옵티마이저 타입"
    )
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    create_directories(args.checkpoint_dir)
    
    # Accelerator 초기화
    accelerator = Accelerator(mixed_precision="fp16")
    
    # 모델 로드
    accelerator.print("=" * 60)
    accelerator.print("TripoSR 모델 로드 중...")
    accelerator.print("=" * 60)
    model, device = load_tripodsr_model()
    
    # LoRA 적용
    accelerator.print("\n" + "=" * 60)
    accelerator.print("LoRA 적용 중...")
    accelerator.print("=" * 60)
    model = apply_lora_to_model(model, r=args.r, alpha=args.alpha)
    
    # 데이터셋 로드
    accelerator.print("\n" + "=" * 60)
    accelerator.print("데이터셋 로드 중...")
    accelerator.print("=" * 60)
    dataset = ImageDataset(args.data_dir, image_size=args.image_size)
    
    if len(dataset) == 0:
        accelerator.print(f"Error: {args.data_dir}에서 이미지를 찾을 수 없습니다.")
        return
    
    accelerator.print(f"로드된 이미지 수: {len(dataset)}개")
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device != "cpu" else False
    )
    
    # 옵티마이저 설정
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Accelerator로 준비
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # 학습 루프
    accelerator.print("\n" + "=" * 60)
    accelerator.print(f"학습 시작 ({args.num_epochs} epochs)")
    accelerator.print(f"배치 크기: {args.batch_size}, 학습률: {args.learning_rate}")
    accelerator.print(f"Loss 타입: {args.loss_type}, 옵티마이저: {args.optimizer}")
    accelerator.print("=" * 60)
    
    for epoch in range(args.num_epochs):
        accelerator.print(f"\n[Epoch {epoch+1}/{args.num_epochs}]")
        avg_loss = train_epoch(model, dataloader, optimizer, accelerator, device, loss_type=args.loss_type)
        accelerator.print(f"Epoch {epoch+1} 평균 Loss: {avg_loss:.6f}")
    
    # LoRA 가중치 저장
    accelerator.print("\n" + "=" * 60)
    accelerator.print("LoRA 가중치 저장 중...")
    accelerator.print("=" * 60)
    
    unwrapped_model = accelerator.unwrap_model(model)
    output_path = os.path.join(
        args.checkpoint_dir,
        f"triposr_lora_r{args.r}_a{args.alpha}.safetensors"
    )
    save_lora_weights(unwrapped_model, output_path, r=args.r, alpha=args.alpha)
    
    accelerator.print("\n" + "=" * 60)
    accelerator.print("학습 완료!")
    accelerator.print(f"저장 경로: {output_path}")
    accelerator.print("=" * 60)


if __name__ == "__main__":
    main()
