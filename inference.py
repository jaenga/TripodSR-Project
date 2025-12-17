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
# 1) 입력 이미지 전처리: 알파 마스크 기반 bbox 크롭 + 패딩
# -----------------------------
def clean_input_image(
    image: Image.Image,
    alpha_threshold: int = 128,
    pad_ratio: float = 0.2,
    bg: str = "white"
) -> Image.Image:
    """입력 이미지를 전처리합니다.
    
    Args:
        image: 입력 이미지 (RGB 또는 RGBA)
        alpha_threshold: 알파 채널 임계값 (기본값: 128)
        pad_ratio: bbox 크롭 후 패딩 비율 (기본값: 0.2)
        bg: 배경 색상 ("white" 또는 "gray", 기본값: "white")
    
    Returns:
        전처리된 RGB 이미지
    """
    # RGBA로 변환
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    img_array = np.array(image)
    alpha = img_array[:, :, 3]
    
    # 알파 채널 임계값 적용하여 마스크 생성
    mask = (alpha > alpha_threshold).astype(np.uint8)
    
    # 마스크가 모두 0이면 전체 이미지 사용
    if np.sum(mask) == 0:
        mask = np.ones_like(alpha, dtype=np.uint8)
    
    # bbox 찾기
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        # 마스크가 없으면 전체 이미지 사용
        y_min, x_min = 0, 0
        y_max, x_max = image.height, image.width
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
    
    # bbox 크롭
    cropped = image.crop((x_min, y_min, x_max, y_max))
    cropped_array = np.array(cropped)
    cropped_alpha = cropped_array[:, :, 3] if cropped.mode == "RGBA" else np.ones((cropped.height, cropped.width), dtype=np.uint8) * 255
    
    # 패딩 계산
    width = x_max - x_min
    height = y_max - y_min
    pad_w = int(width * pad_ratio)
    pad_h = int(height * pad_ratio)
    
    # 정사각형 캔버스 크기 계산
    max_size = max(width + 2 * pad_w, height + 2 * pad_h)
    
    # 배경 색상 결정
    if bg == "gray":
        bg_color = (127, 127, 127, 255)
    else:  # white
        bg_color = (255, 255, 255, 255)
    
    # 새 캔버스 생성
    new_image = Image.new("RGBA", (max_size, max_size), bg_color)
    
    # 크롭된 이미지를 중앙에 배치
    paste_x = (max_size - width) // 2
    paste_y = (max_size - height) // 2
    
    # 알파 마스크 적용하여 붙여넣기
    new_image.paste(cropped, (paste_x, paste_y), cropped.split()[3] if cropped.mode == "RGBA" else None)
    
    # RGB로 변환
    rgb_image = Image.new("RGB", new_image.size, bg_color[:3])
    rgb_image.paste(new_image, mask=new_image.split()[3])
    
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
            # 이미지 로드 및 전처리
            image = Image.open(img_path)
            image = clean_input_image(
                image,
                alpha_threshold=args.alpha_threshold,
                pad_ratio=args.pad_ratio,
                bg=args.bg
            )
            
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
