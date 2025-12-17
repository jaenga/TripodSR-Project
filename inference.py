import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Callable, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import trimesh

from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import load_file

# TripoSR 백본 로드
try:
    from triposr_backbone import load_tripodsr_model as load_model
except ImportError as e:
    raise ImportError("triposr_backbone를 찾을 수 없습니다. 프로젝트 루트/환경을 확인하세요.") from e


# -----------------------------
# 1) 입력 이미지 전처리: bbox 크롭 + 중앙 정렬 + 512x512 리사이즈
# -----------------------------
def bbox_crop_center_pad(image: Image.Image) -> Image.Image:
    """입력 이미지를 객체 중심 정렬하여 전처리합니다.
    
    TripoSR 공식 파이프라인과 동일하게 객체를 중앙에 배치하여
    배경 플레인(판막) 아티팩트를 방지합니다.
    
    Args:
        image: 입력 이미지 (RGB 또는 RGBA)
    
    Returns:
        전처리된 RGB 이미지 (512x512, 흰 배경)
    """
    # RGBA로 변환
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    img_array = np.array(image)
    alpha = img_array[:, :, 3]
    
    # 1. 마스크 생성: 알파 채널이 있으면 알파 기준, 없으면 배경과 색상 차이로
    mask = None
    
    # 알파 채널이 유효한 경우 (일부라도 투명도가 있으면)
    if np.any(alpha < 255):
        # 알파 채널 기준 마스크 생성
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        # 알파가 없거나 모두 불투명한 경우: 배경과 색상 차이로 마스크 생성
        rgb = img_array[:, :, :3]
        
        # 가장자리 픽셀들의 평균 색상 (배경 추정)
        h, w = rgb.shape[:2]
        edge_pixels = np.concatenate([
            rgb[0, :].reshape(-1, 3),  # 상단
            rgb[-1, :].reshape(-1, 3),  # 하단
            rgb[:, 0].reshape(-1, 3),   # 좌측
            rgb[:, -1].reshape(-1, 3)   # 우측
        ], axis=0)
        bg_color = np.median(edge_pixels, axis=0).astype(np.float32)
        
        # 배경과의 색상 차이 계산
        rgb_float = rgb.astype(np.float32)
        color_diff = np.linalg.norm(rgb_float - bg_color, axis=2)
        
        # 임계값: 배경과 차이가 큰 픽셀을 객체로 판단
        threshold = np.percentile(color_diff, 10) + (np.percentile(color_diff, 90) - np.percentile(color_diff, 10)) * 0.3
        mask = (color_diff > threshold).astype(np.uint8) * 255
    
    # 마스크가 모두 0이면 전체 이미지 사용
    if mask is None or np.sum(mask) == 0:
        mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
    
    # 2. bbox 찾기
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        # 마스크가 없으면 전체 이미지 사용
        y_min, x_min = 0, 0
        y_max, x_max = image.height, image.width
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
    
    # 3. bbox 기준 크롭
    cropped = image.crop((x_min, y_min, x_max, y_max))
    
    # 4. 정사각형 캔버스에 중앙 배치
    width = x_max - x_min
    height = y_max - y_min
    max_size = max(width, height)
    
    # 정사각형 캔버스 생성 (흰 배경)
    square_image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    
    # 크롭된 이미지를 중앙에 배치
    paste_x = (max_size - width) // 2
    paste_y = (max_size - height) // 2
    
    # 알파 마스크 적용하여 붙여넣기
    if cropped.mode == "RGBA":
        square_image.paste(cropped, (paste_x, paste_y), cropped.split()[3])
    else:
        square_image.paste(cropped, (paste_x, paste_y))
    
    # 5. 512x512로 리사이즈
    square_image = square_image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 6. 흰 배경 합성 후 RGB로 변환
    rgb_image = Image.new("RGB", square_image.size, (255, 255, 255))
    if square_image.mode == "RGBA":
        rgb_image.paste(square_image, mask=square_image.split()[3])
    else:
        rgb_image.paste(square_image)
    
    return rgb_image


# -----------------------------
# 2) 메쉬 정제: OOB face 제거 + 재인덱싱 + degenerate face 제거
# -----------------------------
def fix_mesh_for_glb(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """메쉬를 GLB 내보내기에 적합하게 정제합니다.
    
    Args:
        mesh: 입력 메쉬
    
    Returns:
        정제된 메쉬
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    if vertices.size == 0 or faces.size == 0:
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    v_count = len(vertices)
    
    # 1) OOB face 제거
    valid_face_mask = np.all((faces >= 0) & (faces < v_count), axis=1)
    faces = faces[valid_face_mask]
    if len(faces) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    # 2) vertex colors 추출
    colors: Optional[np.ndarray] = None
    visual_obj = getattr(mesh, "visual", None)
    if visual_obj is not None:
        vc = getattr(visual_obj, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if len(vc_arr) == v_count:
                colors = vc_arr
    
    # 3) 사용되는 정점만 찾기
    used_vertices = np.unique(faces.reshape(-1))
    used_vertices = used_vertices[(used_vertices >= 0) & (used_vertices < v_count)]
    
    if len(used_vertices) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    # 4) 정점 재인덱싱
    new_vertices = vertices[used_vertices]
    index_map = np.full(v_count, -1, dtype=np.int64)
    index_map[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    
    new_faces = index_map[faces]
    valid_face_mask2 = np.all(new_faces >= 0, axis=1)
    new_faces = new_faces[valid_face_mask2]
    
    if len(new_faces) == 0:
        return trimesh.Trimesh(vertices=new_vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    # 5) degenerate face 제거 (중복 인덱스 또는 면적 0)
    degenerate_mask = np.ones(len(new_faces), dtype=bool)
    for i, face in enumerate(new_faces):
        # 중복 인덱스 체크
        if len(np.unique(face)) < 3:
            degenerate_mask[i] = False
            continue
        
        # 면적 체크
        v0, v1, v2 = new_vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        if area < 1e-10:
            degenerate_mask[i] = False
    
    new_faces = new_faces[degenerate_mask]
    
    if len(new_faces) == 0:
        return trimesh.Trimesh(vertices=new_vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    # 6) vertex colors 재매핑
    new_colors: Optional[np.ndarray] = None
    if colors is not None:
        new_colors = colors[used_vertices]
    
    # 7) 새 메쉬 생성
    fixed = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
    
    # 8) vertex colors 설정
    if new_colors is not None:
        vis2 = getattr(fixed, "visual", None)
        if vis2 is not None:
            try:
                setattr(vis2, "vertex_colors", new_colors)
            except Exception:
                pass
    
    return fixed


# -----------------------------
# 3) LoRA 로드/병합
# -----------------------------
def load_lora_weights(model: nn.Module, lora_path: str, device: torch.device) -> nn.Module:
    """LoRA 가중치를 로드하고 병합합니다.
    
    Args:
        model: 베이스 모델
        lora_path: LoRA 가중치 파일 경로
        device: 디바이스
    
    Returns:
        LoRA가 병합된 모델
    """
    print(f"LoRA 가중치 로드 중: {lora_path}")
    
    try:
        lora_state = load_file(lora_path)
    except Exception as e:
        print(f"Warning: LoRA 파일 로드 실패: {e}")
        return model
    
    # target_modules 탐색 (attn 포함 linear/conv만)
    target_modules: list[str] = []
    for name, module in model.named_modules():
        if "attn" in name.lower() and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            target_modules.append(name)
    
    if not target_modules:
        print("Warning: LoRA target_modules를 찾지 못했습니다. 베이스 모델 사용.")
        return model
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    
    try:
        # pyright 타입 에러 방지
        peft_model_any = cast(Any, get_peft_model(cast(Any, model), lora_config))  # type: ignore
        missing, unexpected = peft_model_any.load_state_dict(lora_state, strict=False)
        
        if missing:
            print(f"Warning: missing keys (sample): {list(missing)[:5]}")
        if unexpected:
            print(f"Warning: unexpected keys (sample): {list(unexpected)[:5]}")
        
        print("LoRA 병합 중...")
        merged: nn.Module = peft_model_any.merge_and_unload()
        merged = merged.to(device).eval()
        return merged
    except Exception as e:
        print(f"Warning: LoRA 로드/병합 실패: {e}. 베이스 모델 사용.")
        return model


# -----------------------------
# 4) 카테고리 맵 로드
# -----------------------------
def load_category_map(path: str) -> Dict[str, Dict[str, Any]]:
    """카테고리 맵을 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["image_name"]: entry for entry in data}


def find_original_name(category_map: Dict[str, Dict[str, Any]], img_path: Path) -> Optional[str]:
    """이미지 파일명에서 원본 이름을 찾습니다."""
    name = img_path.name
    base = name.replace("_no_bg.png", "").replace("_no_bg.PNG", "")
    base_no_ext = Path(base).stem
    
    for ext in [".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG"]:
        cand = base_no_ext + ext
        if cand in category_map:
            return cand
    
    # stem 부분 매칭
    for k in category_map.keys():
        if base_no_ext == Path(k).stem:
            return k
    return None


# -----------------------------
# 5) 메인 함수
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="TripoSR 3D 모델 생성")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw_images/no_background",
        help="입력 이미지 디렉토리 (기본값: data/raw_images/no_background)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/glb_models",
        help="출력 디렉토리 (기본값: outputs/glb_models)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="LoRA 가중치를 사용하지 않고 베이스 모델만 사용"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="checkpoints/lora_weights.safetensors",
        help="LoRA 가중치 파일 경로 (기본값: checkpoints/lora_weights.safetensors)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="메쉬 해상도 (기본값: 256)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="메쉬 추출 임계값 (기본값: 25.0)"
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=128,
        help="RGBA 입력일 때 알파 임계값 (기본값: 128)"
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.2,
        help="bbox 크롭 후 패딩 비율 (기본값: 0.2)"
    )
    parser.add_argument(
        "--bg",
        type=str,
        default="white",
        choices=["white", "gray"],
        help="배경 색상: white 또는 gray (기본값: white)"
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 로드
    print("베이스 모델 로드 중...")
    base_model, _ = load_model(device=str(device))
    base_model = cast(nn.Module, base_model).to(device).eval()
    
    # LoRA 로드
    if args.no_lora:
        print("⚠ LoRA 비활성화 → 베이스 모델만 사용")
        model = base_model
    elif os.path.exists(args.lora_path):
        print("✓ LoRA 가중치 로드 중...")
        model = load_lora_weights(base_model, args.lora_path, device)
    else:
        print(f"⚠ LoRA 없음 → 베이스 모델 사용: {args.lora_path}")
        model = base_model
    
    # 모델 호출 준비
    model_call = cast(Callable[..., Any], model)
    
    # extract_mesh 메서드 확인
    if not hasattr(model, "extract_mesh"):
        raise AttributeError("모델에 extract_mesh 메서드가 없습니다.")
    extract_mesh = cast(Callable[..., Any], getattr(model, "extract_mesh"))
    
    # 카테고리 맵 로드
    category_map_path = "data/image_category_map.json"
    category_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(category_map_path):
        category_map = load_category_map(category_map_path)
    else:
        print(f"Warning: 카테고리 맵 파일 없음: {category_map_path}")
    
    # 이미지 경로 수집
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
    
    image_paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        image_paths.extend(input_dir.glob(ext))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        raise FileNotFoundError(f"처리할 이미지가 없습니다: {input_dir}")
    
    print(f"\n처리할 이미지: {len(image_paths)}개")
    print(f"출력 디렉토리: {output_dir}\n")
    
    success_count = 0
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] ▶ {img_path.name}")
        
        # 카테고리 정보 출력 (있으면)
        original_name = find_original_name(category_map, img_path)
        if original_name and original_name in category_map:
            info = category_map[original_name]
            print(f"  - category: {info.get('category')} (conf={info.get('confidence')})")
        
        try:
            # 이미지 로드 및 전처리 (bbox 크롭 + 중앙 정렬 + 512x512)
            image = Image.open(img_path)
            image = bbox_crop_center_pad(image)
            
            # 3D 생성
            with torch.no_grad():
                scene_codes = model_call(image, device=str(device))
                meshes = extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=args.resolution,
                    threshold=args.threshold,
                )
            
            if not meshes:
                print("  - mesh 없음 → skip")
                continue
            
            mesh0 = meshes[0]
            if not isinstance(mesh0, trimesh.Trimesh):
                # open3d mesh 변환 시도
                try:
                    v = np.asarray(getattr(mesh0, "vertices"))
                    f = np.asarray(getattr(mesh0, "triangles"))
                    mesh0 = trimesh.Trimesh(vertices=v, faces=f, process=False)
                except Exception:
                    print(f"  - mesh 타입 변환 실패: {type(mesh0)}")
                    continue
            
            # 메쉬 정제
            fixed = fix_mesh_for_glb(mesh0)
            
            # GLB 내보내기
            out_file = output_dir / f"{img_path.stem}.glb"
            scene = trimesh.Scene([fixed])
            scene.export(str(out_file), file_type="glb")
            
            print(f"  ✅ saved: {out_file}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n완료: {success_count}/{len(image_paths)}개 처리 성공")


if __name__ == "__main__":
    main()
