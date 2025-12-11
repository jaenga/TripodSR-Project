# pip install torch torchvision
# pip install peft safetensors
# pip install Pillow
# pip install trimesh

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from safetensors.torch import load_file
from triposr_backbone import load_tripodsr_model
from train_lora import apply_lora_to_model

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh가 설치되지 않았습니다. GLTF 변환에 문제가 있을 수 있습니다.")


def create_directories():
    """필요한 디렉토리가 없으면 생성합니다."""
    os.makedirs("outputs/gltf_models/baseline", exist_ok=True)
    os.makedirs("outputs/gltf_models/lora", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw_images", exist_ok=True)


def preprocess_image_for_triposr(
    image_path: str,
    image_size: int = 256,
    center_crop: bool = True
) -> Image.Image:
    """이미지를 TripoSR에 맞게 전처리합니다.
    
    Args:
        image_path: 이미지 파일 경로
        image_size: 리사이즈할 이미지 크기
        center_crop: True면 center crop, False면 그냥 resize
    
    Returns:
        전처리된 PIL Image (RGB)
    """
    image = Image.open(image_path).convert("RGB")
    
    if center_crop:
        # 비율을 유지하면서 center crop
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
    
    # 리사이즈
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return image


def load_lora_metadata(lora_path: str) -> Optional[Dict]:
    """LoRA 메타데이터 파일을 읽어서 r, alpha 값을 반환합니다.
    
    Args:
        lora_path: LoRA 가중치 파일 경로 (.safetensors)
    
    Returns:
        메타데이터 딕셔너리 (r, alpha 포함) 또는 None (파일이 없으면)
    """
    # 메타데이터 파일 경로: .safetensors를 _config.json으로 변경
    metadata_path = lora_path.replace(".safetensors", "_config.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Warning: 메타데이터 파일 읽기 실패: {e}")
            return None
    else:
        return None


def load_lora_weights(
    model: nn.Module,
    lora_path: str,
    device: str,
    r: Optional[int] = None,
    alpha: Optional[int] = None
):
    """LoRA 가중치를 로드하고 모델에 적용합니다.
    
    Args:
        model: 베이스 모델
        lora_path: LoRA 가중치 파일 경로
        device: 디바이스 문자열
        r: LoRA rank (None이면 메타데이터에서 읽거나 기본값 사용)
        alpha: LoRA alpha (None이면 메타데이터에서 읽거나 기본값 사용)
    
    Returns:
        LoRA가 병합된 모델
    """
    print(f"LoRA 가중치 로드 중: {lora_path}")
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA 가중치 파일을 찾을 수 없습니다: {lora_path}")
    
    # 메타데이터에서 r, alpha 읽기 시도
    metadata = load_lora_metadata(lora_path)
    
    if metadata is not None:
        metadata_r = metadata.get("r")
        metadata_alpha = metadata.get("alpha")
        
        if r is None:
            r = metadata_r
            print(f"메타데이터에서 r={r} 읽기 완료")
        elif r != metadata_r:
            print(f"Warning: 인자로 받은 r={r}와 메타데이터의 r={metadata_r}가 다릅니다. 인자 값을 사용합니다.")
        
        if alpha is None:
            alpha = metadata_alpha
            print(f"메타데이터에서 alpha={alpha} 읽기 완료")
        elif alpha != metadata_alpha:
            print(f"Warning: 인자로 받은 alpha={alpha}와 메타데이터의 alpha={metadata_alpha}가 다릅니다. 인자 값을 사용합니다.")
    else:
        # 메타데이터가 없으면 기본값 사용 (경고 출력)
        if r is None:
            r = 4
            print(f"Warning: LoRA 메타데이터를 찾을 수 없습니다. 기본값 r={r}를 사용합니다.")
        if alpha is None:
            alpha = 32
            print(f"Warning: LoRA 메타데이터를 찾을 수 없습니다. 기본값 alpha={alpha}를 사용합니다.")
    
    # LoRA 가중치 로드
    lora_state_dict = load_file(lora_path)
    
    # apply_lora_to_model을 사용하여 LoRA 적용 (train_lora.py와 동일한 로직)
    print(f"LoRA 적용 중 (r={r}, alpha={alpha})...")
    lora_model = apply_lora_to_model(model, r=r, alpha=alpha)
    
    # LoRA 가중치 로드
    model_state_dict = lora_model.state_dict()
    loaded_state_dict = {}
    
    for key, value in lora_state_dict.items():
        if key in model_state_dict:
            loaded_state_dict[key] = value
        else:
            # 키 이름이 다를 수 있으므로 유사한 키 찾기
            found = False
            for model_key in model_state_dict.keys():
                if "lora" in model_key.lower() and key.split(".")[-1] == model_key.split(".")[-1]:
                    loaded_state_dict[model_key] = value
                    found = True
                    break
            if not found:
                print(f"Warning: LoRA 키 '{key}'를 모델에 매칭할 수 없습니다.")
    
    # 가중치 로드
    missing_keys, unexpected_keys = lora_model.load_state_dict(loaded_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: 누락된 키: {len(missing_keys)}개 (처음 5개: {missing_keys[:5]})")
    if unexpected_keys:
        print(f"Warning: 예상치 못한 키: {len(unexpected_keys)}개 (처음 5개: {unexpected_keys[:5]})")
    
    # LoRA 병합: 어댑터 가중치를 베이스 모델에 병합
    print("LoRA 가중치를 베이스 모델에 병합 중...")
    merged_model = lora_model.merge_and_unload()
    merged_model = merged_model.to(device)
    merged_model.eval()
    
    print("LoRA 병합 완료!")
    return merged_model


def load_image_category_map(json_path: str) -> Dict[str, Dict]:
    """이미지-카테고리 매핑 JSON 파일을 로드합니다.
    
    Args:
        json_path: JSON 파일 경로
    
    Returns:
        이미지 이름을 키로 하는 딕셔너리
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 리스트를 딕셔너리로 변환
    category_map = {}
    for entry in data:
        image_name = entry["image_name"]
        category_map[image_name] = {
            "category": entry["category"],
            "confidence": entry["confidence"]
        }
    
    return category_map


def generate_3d_mesh(
    model,
    image: Image.Image,
    device: str,
    mc_resolution: int = 256,
    has_vertex_color: bool = True
):
    """TripoSR 모델을 사용하여 3D 메쉬를 생성합니다.
    
    Args:
        model: TripoSR 모델 인스턴스
        image: PIL Image (RGB)
        device: 디바이스 문자열
        mc_resolution: Marching Cubes 해상도
        has_vertex_color: True면 vertex color 포함, False면 텍스처 없음
    
    Returns:
        trimesh.Trimesh 객체
    """
    with torch.no_grad():
        # TripoSR forward: PIL Image를 받아서 scene_codes 생성
        scene_codes = model([image], device=device)
        
        # 메쉬 추출
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=has_vertex_color,
            resolution=mc_resolution
        )
        
        # 첫 번째 메쉬 반환 (배치가 1이므로)
        return meshes[0]


def mesh_to_gltf(mesh, output_path: str):
    """메쉬를 GLTF 형식으로 변환하여 저장합니다.
    
    Args:
        mesh: trimesh.Trimesh 객체
        output_path: 출력 GLTF 파일 경로
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh가 설치되지 않았습니다.")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # trimesh는 직접 GLTF로 내보낼 수 있음
    if isinstance(mesh, trimesh.Trimesh):
        mesh.export(output_path, file_type="gltf")
        print(f"GLTF 파일 저장 완료: {output_path}")
    else:
        raise ValueError(f"지원되지 않는 메쉬 타입: {type(mesh)}")


def process_images(
    model,
    device: str,
    category_map: Dict[str, Dict],
    image_paths: List[Path],
    output_dir: str,
    mc_resolution: int = 256,
    max_samples: Optional[int] = None
):
    """이미지들을 처리하여 3D 메쉬를 생성하고 저장합니다.
    
    Args:
        model: TripoSR 모델 인스턴스
        device: 디바이스 문자열
        category_map: 이미지-카테고리 매핑 딕셔너리
        image_paths: 처리할 이미지 경로 리스트
        output_dir: 출력 디렉토리 (baseline 또는 lora)
        mc_resolution: Marching Cubes 해상도
        max_samples: 최대 처리할 이미지 개수 (None이면 모두 처리)
    """
    if max_samples is not None:
        image_paths = image_paths[:max_samples]
    
    print(f"처리할 이미지 수: {len(image_paths)}개")
    
    for idx, image_path in enumerate(image_paths):
        image_name = image_path.name
        print(f"\n[{idx + 1}/{len(image_paths)}] 처리 중: {image_name}")
        
        # 카테고리 확인
        if image_name not in category_map:
            print(f"Warning: {image_name}에 대한 카테고리가 없습니다. 건너뜁니다.")
            continue
        
        category_info = category_map[image_name]
        category = category_info["category"]
        confidence = category_info["confidence"]
        
        print(f"  카테고리: {category} (신뢰도: {confidence:.4f})")
        
        try:
            # 이미지 전처리
            print("  이미지 전처리 중...")
            processed_image = preprocess_image_for_triposr(str(image_path))
            
            # 3D 메쉬 생성
            print("  3D 메쉬 생성 중...")
            mesh = generate_3d_mesh(
                model,
                processed_image,
                device,
                mc_resolution=mc_resolution,
                has_vertex_color=True
            )
            
            # GLTF 형식으로 저장
            output_path = os.path.join(output_dir, f"{image_path.stem}.gltf")
            print(f"  GLTF 파일로 저장 중: {output_path}")
            mesh_to_gltf(mesh, output_path)
            
        except Exception as e:
            print(f"  Error: 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """메인 추론 함수"""
    parser = argparse.ArgumentParser(description="TripoSR을 사용한 3D 모델 생성")
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="LoRA 가중치를 사용하여 모델을 로드합니다"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors",
        help="LoRA 가중치 파일 경로"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank (None이면 메타데이터에서 읽거나 기본값 사용)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (None이면 메타데이터에서 읽거나 기본값 사용)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="처리할 최대 이미지 개수 (None이면 모두 처리)"
    )
    parser.add_argument(
        "--mc_resolution",
        type=int,
        default=256,
        help="Marching Cubes 해상도 (기본값: 256)"
    )
    parser.add_argument(
        "--category_map",
        type=str,
        default="data/image_category_map.json",
        help="이미지-카테고리 매핑 JSON 파일 경로"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/raw_images",
        help="입력 이미지 디렉토리"
    )
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    create_directories()
    
    # 베이스 모델 로드
    print("=" * 60)
    print("TripoSR 베이스 모델 로드 중...")
    print("=" * 60)
    base_model, device = load_tripodsr_model()
    
    # LoRA 가중치 로드 (옵션)
    if args.use_lora:
        print("\n" + "=" * 60)
        print("LoRA 가중치 로드 및 병합 중...")
        print("=" * 60)
        model = load_lora_weights(
            base_model,
            args.lora_path,
            device,
            r=args.lora_r,
            alpha=args.lora_alpha
        )
        output_subdir = "lora"
    else:
        print("\n베이스 모델만 사용합니다.")
        model = base_model
        output_subdir = "baseline"
    
    # 이미지-카테고리 매핑 로드
    print("\n" + "=" * 60)
    print("카테고리 맵 로드 중...")
    print("=" * 60)
    if not os.path.exists(args.category_map):
        print(f"Error: 카테고리 맵 파일을 찾을 수 없습니다: {args.category_map}")
        return
    
    category_map = load_image_category_map(args.category_map)
    print(f"로드된 이미지-카테고리 매핑: {len(category_map)}개")
    
    # 이미지 디렉토리에서 모든 이미지 로드
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + \
                  list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG"))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"Error: {image_dir}에서 이미지를 찾을 수 없습니다.")
        return
    
    # 출력 디렉토리 설정
    output_dir = os.path.join("outputs/gltf_models", output_subdir)
    
    # 이미지 처리
    print("\n" + "=" * 60)
    print(f"3D 모델 생성 시작 ({output_subdir} 모델 사용)")
    print("=" * 60)
    process_images(
        model=model,
        device=device,
        category_map=category_map,
        image_paths=image_paths,
        output_dir=output_dir,
        mc_resolution=args.mc_resolution,
        max_samples=args.max_samples
    )
    
    print("\n" + "=" * 60)
    print("모든 추론 작업 완료!")
    print(f"결과 저장 위치: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
