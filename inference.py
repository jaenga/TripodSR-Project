import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Callable, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import trimesh

from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import load_file

# TripoSR / TripodSR 백본 로드 (원본 환경 유지)
try:
    from triposr_backbone import load_tripodsr_model as load_model
except ImportError as e:
    raise ImportError("triposr_backbone를 찾을 수 없습니다. 프로젝트 루트/환경을 확인하세요.") from e


# -----------------------------
# 1) 메쉬 정제: OOB face 제거 + 사용 정점만 남기기 + face 재매핑 + vertex colors 동기화
#    (pyright 경고 방지: getattr/setattr + hasattr로만 접근)
# -----------------------------
def fix_mesh_for_glb(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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

    # 2) vertex colors 안전 추출 (타입스텁 이슈 때문에 무조건 getattr/setattr)
    colors: Optional[np.ndarray] = None
    visual_obj = getattr(mesh, "visual", None)
    if visual_obj is not None:
        vc = getattr(visual_obj, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if len(vc_arr) == v_count:
                colors = vc_arr

    # 3) 사용되는 정점만 남기기
    used = np.unique(faces.reshape(-1))
    used = used[(used >= 0) & (used < v_count)]
    if len(used) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)

    new_vertices = vertices[used]

    # 4) old -> new 인덱스 매핑
    index_map = np.full(v_count, -1, dtype=np.int64)
    index_map[used] = np.arange(len(used), dtype=np.int64)

    new_faces = index_map[faces]
    valid_face_mask2 = np.all(new_faces >= 0, axis=1)
    new_faces = new_faces[valid_face_mask2]
    if len(new_faces) == 0:
        return trimesh.Trimesh(vertices=new_vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)

    # 5) colors도 같이 줄이기
    new_colors: Optional[np.ndarray] = None
    if colors is not None:
        new_colors = colors[used]

    fixed = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True, validate=True)

    # 6) 색상 붙이기 (getattr/setattr)
    if new_colors is not None:
        vis2 = getattr(fixed, "visual", None)
        if vis2 is not None:
            try:
                setattr(vis2, "vertex_colors", new_colors)
            except Exception:
                pass

    # 7) trimesh 메서드 타입스텁 누락 대응: hasattr로만 호출
    if hasattr(fixed, "remove_degenerate_faces"):
        try:
            getattr(fixed, "remove_degenerate_faces")()
        except Exception:
            pass

    if hasattr(fixed, "remove_duplicate_faces"):
        try:
            getattr(fixed, "remove_duplicate_faces")()
        except Exception:
            pass

    if hasattr(fixed, "remove_infinite_values"):
        try:
            getattr(fixed, "remove_infinite_values")()
        except Exception:
            pass

    # 참고: remove_unreferenced_vertices는 process=True면 대부분 해결됨.
    return fixed


# -----------------------------
# 2) LoRA 로드/병합 (pyright: get_peft_model 타입 기대치 문제 -> Any 처리)
# -----------------------------
def load_lora_weights(model: nn.Module, lora_path: str, device: torch.device) -> nn.Module:
    print(f"LoRA 가중치 로드 중: {lora_path}")
    lora_state = load_file(lora_path)

    target_modules: list[str] = []
    for name, module in model.named_modules():
        if "attn" in name.lower() and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            target_modules.append(name)

    if not target_modules:
        for name, module in model.named_modules():
            if "attn" in name.lower() and len(list(module.children())) == 0:
                target_modules.append(name)

    if not target_modules:
        raise ValueError("LoRA target_modules를 찾지 못했습니다. 학습 시 target_modules 설정과 동일한지 확인하세요.")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )

    # pyright 회피: get_peft_model의 타입이 PreTrainedModel로 잡혀있어 nn.Module에 경고 뜸
    peft_model_any = cast(Any, get_peft_model(model, lora_config))  # type: ignore
    missing, unexpected = peft_model_any.load_state_dict(lora_state, strict=False)
    if missing:
        print(f"Warning: missing keys (sample): {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys (sample): {unexpected[:5]}")

    print("LoRA 병합 중...")
    merged: nn.Module = peft_model_any.merge_and_unload()
    merged = merged.to(device).eval()
    return merged


# -----------------------------
# 3) 카테고리 맵 로드 + 파일명 매칭
# -----------------------------
def load_category_map(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["image_name"]: entry for entry in data}


def find_original_name(category_map: Dict[str, Dict[str, Any]], img_path: Path) -> Optional[str]:
    name = img_path.name
    base = name.replace("_no_bg.png", "").replace("_no_bg.PNG", "")
    base_no_ext = Path(base).stem

    for ext in [".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG"]:
        cand = base_no_ext + ext
        if cand in category_map:
            return cand

    # 최후의 수단: stem 부분 매칭
    for k in category_map.keys():
        if base_no_ext == Path(k).stem:
            return k
    return None


# -----------------------------
# 4) main
# -----------------------------
def main() -> None:
    os.makedirs("outputs/glb_models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 모델 로드
    base_model, _ = load_model(device=str(device))
    base_model = cast(nn.Module, base_model).to(device).eval()

    lora_path = "checkpoints/lora_weights.safetensors"
    if os.path.exists(lora_path):
        model: nn.Module = load_lora_weights(base_model, lora_path, device)
    else:
        print(f"Warning: LoRA 없음 → 베이스 모델 사용: {lora_path}")
        model = base_model

    # TripoSR 커스텀 호출을 pyright가 몰라서 Callable[Any]로 캐스팅
    model_call = cast(Callable[..., Any], model)
    
    # extract_mesh 메서드 확인
    if not hasattr(model, "extract_mesh"):
        raise AttributeError("모델에 extract_mesh 메서드가 없습니다. TripoSR 모델이 제대로 로드되었는지 확인하세요.")
    extract_mesh = cast(Callable[..., Any], getattr(model, "extract_mesh"))

    # category map
    category_map_path = "data/image_category_map.json"
    if not os.path.exists(category_map_path):
        raise FileNotFoundError(f"카테고리 맵 파일 없음: {category_map_path}")
    category_map = load_category_map(category_map_path)

    # 이미지 경로
    raw_dir = Path("data/raw_images")
    no_bg_dir = raw_dir / "no_background"
    if no_bg_dir.exists():
        image_paths = sorted(list(no_bg_dir.glob("*.png")) + list(no_bg_dir.glob("*.PNG")))
    else:
        image_paths = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            image_paths.extend(raw_dir.glob(ext))
        image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError("처리할 이미지가 없습니다. data/raw_images 또는 data/raw_images/no_background 확인")

    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] ▶ {img_path.name}")

        original_name = find_original_name(category_map, img_path)
        if original_name is None:
            print("  - category 매칭 실패 → skip")
            continue

        info = category_map[original_name]
        print(f"  - category: {info.get('category')} (conf={info.get('confidence')})")

        # 이미지 로드 (RGBA -> 흰배경 합성)
        image = Image.open(img_path)
        if image.mode == "RGBA":
            rgb = Image.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[3])
            image = rgb
        else:
            image = image.convert("RGB")

        with torch.no_grad():
            try:
                scene_codes = model_call(image, device=str(device))
                meshes = extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=256,
                    threshold=25.0,
                )
            except Exception as e:
                print(f"  - Error: 추론 실패: {e}")
                continue

        if not meshes:
            print("  - mesh 없음 → skip")
            continue

        mesh0 = meshes[0]
        if not isinstance(mesh0, trimesh.Trimesh):
            # 혹시 open3d mesh면 변환 시도
            try:
                v = np.asarray(getattr(mesh0, "vertices"))
                f = np.asarray(getattr(mesh0, "triangles"))
                mesh0 = trimesh.Trimesh(vertices=v, faces=f, process=False)
            except Exception:
                print(f"  - mesh 타입 변환 실패: {type(mesh0)}")
                continue

        fixed = fix_mesh_for_glb(mesh0)

        out_file = Path("outputs/glb_models") / f"{img_path.stem}.glb"
        try:
            fixed.export(str(out_file), file_type="glb")
            print(f"  ✅ saved: {out_file}")
        except Exception as e:
            print(f"  - Error: GLB export 실패: {e}")


if __name__ == "__main__":
    main()