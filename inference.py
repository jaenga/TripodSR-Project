# pip install torch torchvision
# pip install peft safetensors
# pip install Pillow
# pip install open3d
# pip install trimesh

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel
from safetensors.torch import load_file
import numpy as np

# Colab 환경 감지
def is_colab():
    """Colab 환경인지 확인"""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False

# Colab 환경에 따른 경로 설정
if is_colab():
    DEFAULT_LORA_PATH = "/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors"
else:
    DEFAULT_LORA_PATH = "checkpoints/lora_weights.safetensors"

try:
    import open3d as o3d
    import trimesh
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d 또는 trimesh가 설치되지 않았습니다. GLTF 변환에 문제가 있을 수 있습니다.")

def create_directories():
    """필요한 디렉토리가 없으면 생성합니다."""
    os.makedirs("outputs/gltf_models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw_images", exist_ok=True)

def get_device():
    """CUDA가 사용 가능하면 GPU를, 아니면 CPU를 반환합니다."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA 사용 불가. CPU를 사용합니다.")
    return device

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

def load_lora_weights(model: nn.Module, lora_path: str, device: torch.device):
    """LoRA 가중치를 로드하고 모델에 적용합니다.
    
    Args:
        model: 베이스 모델
        lora_path: LoRA 가중치 파일 경로
        device: 디바이스
    
    Returns:
        LoRA가 병합된 모델
    """
    print(f"LoRA 가중치 로드 중: {lora_path}")
    
    # LoRA 가중치 로드
    lora_state_dict = load_file(lora_path)
    
    # PEFT를 사용하여 LoRA 어댑터 적용
    # 먼저 PEFT 모델로 래핑
    from peft import LoraConfig, TaskType, get_peft_model
    
    # LoRA 설정 (학습 시와 동일한 설정 사용)
    # train_lora.py의 apply_lora_to_model과 동일한 target_modules 사용
    target_modules = []
    for name, module in model.named_modules():
        if "attn" in name.lower():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                target_modules.append(name)
    
    if not target_modules:
        # Fallback: leaf modules with 'attn' in name
        for name, module in model.named_modules():
            if "attn" in name.lower() and len(list(module.children())) == 0:
                target_modules.append(name)
    
    if not target_modules:
        raise ValueError("No modules with 'attn' in name found for LoRA loading")
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,  # rank
        lora_alpha=32,  # alpha
        target_modules=target_modules,  # 학습 시와 동일한 모듈 사용
        lora_dropout=0.1,
        bias="none",
    )
    
    # LoRA 모델 생성
    lora_model = get_peft_model(model, lora_config)  # type: ignore
    
    # LoRA 가중치 로드
    # 가중치 키 이름을 모델에 맞게 조정
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
        print(f"Warning: 누락된 키: {missing_keys[:5]}...")  # 처음 5개만 출력
    if unexpected_keys:
        print(f"Warning: 예상치 못한 키: {unexpected_keys[:5]}...")
    
    # LoRA 병합: 어댑터 가중치를 베이스 모델에 병합
    print("LoRA 가중치를 베이스 모델에 병합 중...")
    merged_model = lora_model.merge_and_unload()  # type: ignore
    
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

def load_and_preprocess_image(image_path: str, image_size: int = 256) -> Image.Image:
    """이미지를 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로
        image_size: 이미지 크기 (사용하지 않음, TripoSR이 자체 처리)
    
    Returns:
        PIL Image 객체
    """
    image = Image.open(image_path).convert("RGB")
    return image

def fix_mesh_indices(mesh):
    """메쉬의 인덱스 범위 초과 문제를 수정합니다.
    
    Args:
        mesh: trimesh.Trimesh 객체
    
    Returns:
        수정된 trimesh.Trimesh 객체
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    # vertices와 faces 확인
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    if len(faces) == 0:
        return mesh
    
    num_vertices = len(vertices)
    original_face_count = len(faces)
    original_vertex_count = num_vertices
    
    # 반복적으로 수정하여 완전히 고정될 때까지
    max_iterations = 5
    for iteration in range(max_iterations):
        # 1단계: 범위를 벗어나는 인덱스를 가진 face 제거
        # 각 face의 모든 인덱스가 유효한 범위 내에 있는지 확인
        valid_mask = np.all((faces >= 0) & (faces < num_vertices), axis=1)
        
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            if iteration == 0:
                print(f"  Warning: {invalid_count}개의 out-of-bound face 제거 중...")
            faces = faces[valid_mask]
            
            # face가 모두 제거되면 빈 메쉬 반환
            if len(faces) == 0:
                print("  Error: 모든 face가 제거되었습니다. 빈 메쉬를 반환합니다.")
                return trimesh.Trimesh(vertices=vertices, faces=[])
        
        # 2단계: 사용되는 vertex 인덱스만 추출 (범위 체크 포함)
        used_vertex_indices = np.unique(faces.flatten())
        # 범위를 벗어나는 인덱스 제거
        used_vertex_indices = used_vertex_indices[(used_vertex_indices >= 0) & (used_vertex_indices < num_vertices)]
        
        if len(used_vertex_indices) == 0:
            print("  Error: 유효한 vertex 인덱스가 없습니다. 빈 메쉬를 반환합니다.")
            return trimesh.Trimesh(vertices=vertices, faces=[])
        
        # 사용되는 vertices만 추출
        new_vertices = vertices[used_vertex_indices]
        new_num_vertices = len(new_vertices)
        
        # 인덱스 재매핑: old_index -> new_index
        index_map = np.full(num_vertices, -1, dtype=np.int32)  # -1로 초기화하여 매핑되지 않은 인덱스 표시
        index_map[used_vertex_indices] = np.arange(new_num_vertices)
        
        # faces 재인덱싱 (매핑되지 않은 인덱스는 -1이 됨)
        new_faces = index_map[faces]
        
        # 매핑되지 않은 인덱스(-1)를 가진 face 제거
        valid_face_mask = np.all(new_faces >= 0, axis=1)
        if not valid_face_mask.all():
            invalid_face_count = (~valid_face_mask).sum()
            if iteration == 0:
                print(f"  Warning: {invalid_face_count}개의 매핑 실패 face 제거 중...")
            new_faces = new_faces[valid_face_mask]
            
            if len(new_faces) == 0:
                print("  Error: 모든 face가 제거되었습니다. 빈 메쉬를 반환합니다.")
                return trimesh.Trimesh(vertices=new_vertices, faces=[])
        
        # vertex colors가 있으면 재인덱싱
        new_vertex_colors = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:  # type: ignore
            vertex_colors = np.asarray(mesh.visual.vertex_colors)  # type: ignore
            if len(vertex_colors) == num_vertices:
                new_vertex_colors = vertex_colors[used_vertex_indices]
        
        # 최종 검증: 인덱스가 모두 유효한지 확인
        max_index = new_faces.max() if len(new_faces) > 0 else -1
        min_index = new_faces.min() if len(new_faces) > 0 else -1
        
        # 모든 인덱스가 유효한 범위 내에 있으면 완료
        if min_index >= 0 and max_index < new_num_vertices:
            # 새 메쉬 생성
            fixed_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, validate=True, process=True)
            
            if new_vertex_colors is not None:
                fixed_mesh.visual.vertex_colors = new_vertex_colors  # type: ignore
            
            removed_faces = original_face_count - len(new_faces)
            removed_vertices = original_vertex_count - new_num_vertices
            print(f"  ✓ 메쉬 수정 완료 (반복 {iteration + 1}회):")
            print(f"    - Face: {original_face_count} -> {len(new_faces)} (제거: {removed_faces})")
            print(f"    - Vertices: {original_vertex_count} -> {new_num_vertices} (제거: {removed_vertices})")
            print(f"    - 인덱스 범위: [{min_index}, {max_index}] / {new_num_vertices}")
            
            return fixed_mesh
        
        # 아직 문제가 있으면 다음 반복으로
        vertices = new_vertices
        faces = new_faces
        num_vertices = new_num_vertices
    
    # 최대 반복 횟수에 도달했지만 여전히 문제가 있는 경우
    print(f"  ⚠ Warning: 최대 반복 횟수({max_iterations})에 도달했습니다. trimesh.process()로 강제 정리합니다.")
    try:
        # 마지막으로 trimesh.process()로 강제 정리
        final_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, validate=True, process=True)
        if new_vertex_colors is not None:
            final_mesh.visual.vertex_colors = new_vertex_colors  # type: ignore
        return final_mesh
    except Exception as e:
        print(f"  Error: trimesh.process() 실패: {e}")
        # 최후의 수단: 유효한 face만으로 메쉬 생성
        final_faces = faces[(faces >= 0) & (faces < len(vertices))]
        return trimesh.Trimesh(vertices=vertices, faces=final_faces, validate=False)

def fix_gltf_accessors(gltf_path: str):
    """GLTF 파일의 accessor min/max 값과 bufferView target을 실제 데이터에 맞게 수정합니다.
    
    Args:
        gltf_path: GLTF 파일 경로
    """
    import json
    import struct
    
    # GLTF 파일 읽기
    with open(gltf_path, 'r', encoding='utf-8') as f:
        gltf_data = json.load(f)
    
    base_dir = Path(gltf_path).parent
    
    # buffer 데이터 로드
    buffers = {}
    if 'buffers' in gltf_data:
        for i, buffer_info in enumerate(gltf_data['buffers']):
            if 'uri' in buffer_info:
                bin_path = base_dir / buffer_info['uri']
                if bin_path.exists():
                    with open(bin_path, 'rb') as f:
                        buffers[i] = f.read()
                else:
                    buffers[i] = b''
            else:
                buffers[i] = b''
    
    # indices accessor 찾기 (meshes[0].primitives[0].indices에서)
    indices_accessor_set = set()
    if 'meshes' in gltf_data:
        for mesh in gltf_data['meshes']:
            if 'primitives' in mesh:
                for primitive in mesh['primitives']:
                    if 'indices' in primitive:
                        indices_accessor_set.add(primitive['indices'])
    
    # bufferViews와 accessors 수정
    if 'bufferViews' in gltf_data and 'accessors' in gltf_data:
        for accessor_idx, accessor in enumerate(gltf_data['accessors']):
            if 'bufferView' not in accessor:
                continue
            
            buffer_view_idx = accessor['bufferView']
            if buffer_view_idx >= len(gltf_data['bufferViews']):
                continue
            
            buffer_view = gltf_data['bufferViews'][buffer_view_idx]
            buffer_idx = buffer_view.get('buffer', 0)
            
            if buffer_idx not in buffers:
                continue
            
            buffer_data = buffers[buffer_idx]
            byte_offset = buffer_view.get('byteOffset', 0)
            byte_length = buffer_view.get('byteLength', 0)
            
            if byte_offset + byte_length > len(buffer_data):
                continue
            
            # bufferView target 설정
            component_type = accessor.get('componentType', 5123)  # 기본값: UNSIGNED_SHORT
            type_str = accessor.get('type', 'SCALAR')
            
            # indices accessor인 경우 ELEMENT_ARRAY_BUFFER (34963)
            is_indices = accessor_idx in indices_accessor_set
            if is_indices:
                buffer_view['target'] = 34963  # ELEMENT_ARRAY_BUFFER
            else:
                buffer_view['target'] = 34962  # ARRAY_BUFFER
            
            # 실제 데이터 읽기
            data_slice = buffer_data[byte_offset:byte_offset + byte_length]
            
            # componentType에 따른 데이터 타입 결정
            if component_type == 5120:  # BYTE
                dtype = np.int8
            elif component_type == 5121:  # UNSIGNED_BYTE
                dtype = np.uint8
            elif component_type == 5122:  # SHORT
                dtype = np.int16
            elif component_type == 5123:  # UNSIGNED_SHORT
                dtype = np.uint16
            elif component_type == 5125:  # UNSIGNED_INT
                dtype = np.uint32
            elif component_type == 5126:  # FLOAT
                dtype = np.float32
            else:
                continue
            
            # type에 따른 차원 결정
            type_to_count = {
                'SCALAR': 1,
                'VEC2': 2,
                'VEC3': 3,
                'VEC4': 4,
                'MAT2': 4,
                'MAT3': 9,
                'MAT4': 16
            }
            count = type_to_count.get(type_str, 1)
            
            # 데이터 배열로 변환
            try:
                data_array = np.frombuffer(data_slice, dtype=dtype)
                if count > 1:
                    data_array = data_array.reshape(-1, count)
                
                # min/max 계산
                if len(data_array) > 0:
                    min_vals = np.min(data_array, axis=0).tolist()
                    max_vals = np.max(data_array, axis=0).tolist()
                    
                    # accessor 업데이트
                    accessor['min'] = min_vals if count > 1 else [min_vals]
                    accessor['max'] = max_vals if count > 1 else [max_vals]
                    
                    if is_indices:
                        print(f"  ✓ Accessor {accessor_idx} (indices) 수정: min={min_vals}, max={max_vals}")
                    else:
                        print(f"  ✓ Accessor {accessor_idx} ({type_str}) 수정: min={min_vals}, max={max_vals}")
            except Exception as e:
                print(f"  ⚠ Accessor {accessor_idx} 수정 실패: {e}")
                continue
    
    # 버퍼 길이도 수정
    if 'buffers' in gltf_data:
        for i, buffer_info in enumerate(gltf_data['buffers']):
            if 'uri' in buffer_info:
                bin_path = base_dir / buffer_info['uri']
                if bin_path.exists():
                    actual_length = bin_path.stat().st_size
                    buffer_info['byteLength'] = actual_length
    
    # 수정된 GLTF 파일 저장
    with open(gltf_path, 'w', encoding='utf-8') as f:
        json.dump(gltf_data, f, indent=2, ensure_ascii=False)

def fix_gltf_buffer_lengths(gltf_path: str):
    """GLTF 파일의 버퍼 길이 불일치 문제를 수정합니다.
    
    Args:
        gltf_path: GLTF 파일 경로
    """
    # fix_gltf_accessors가 버퍼 길이도 수정하므로 이 함수는 호출만
    fix_gltf_accessors(gltf_path)

def mesh_to_gltf(mesh, output_path: str):
    """메쉬를 GLTF 형식으로 변환하여 저장합니다.
    
    Args:
        mesh: trimesh.Trimesh 객체 또는 open3d 메쉬 객체
        output_path: 출력 GLTF 파일 경로
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d 또는 trimesh가 설치되지 않았습니다.")
    
    # trimesh.Trimesh 객체인 경우 직접 내보내기
    if isinstance(mesh, trimesh.Trimesh):
        # 메쉬 인덱스 문제 수정
        fixed_mesh = fix_mesh_indices(mesh)
        
        # Scene을 사용하여 더 정확하게 내보내기
        scene = trimesh.Scene([fixed_mesh])
        scene.export(output_path, file_type="gltf")
        
        # 버퍼 길이 수정
        print("  GLTF 메타데이터 수정 중...")
        fix_gltf_accessors(output_path)
        print(f"GLTF 파일 저장 완료: {output_path}")
    # open3d 메쉬를 trimesh로 변환
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        # open3d 메쉬를 numpy 배열로 변환
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # vertex colors가 있으면 포함
        vertex_colors = None
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
        
        # trimesh 메쉬 생성
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        
        # 메쉬 인덱스 문제 수정
        fixed_mesh = fix_mesh_indices(tri_mesh)
        
        # Scene을 사용하여 더 정확하게 내보내기
        scene = trimesh.Scene([fixed_mesh])
        scene.export(output_path, file_type="gltf")
        
        # 버퍼 길이 수정
        print("  GLTF 메타데이터 수정 중...")
        fix_gltf_accessors(output_path)
        print(f"GLTF 파일 저장 완료: {output_path}")
    else:
        # 포인트 클라우드인 경우
        if isinstance(mesh, o3d.geometry.PointCloud):
            points = np.asarray(mesh.points)
            # 포인트 클라우드를 간단한 메쉬로 변환 (예: 각 포인트를 작은 구로)
            tri_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
            meshes = []
            for point in points:
                mesh_copy = tri_mesh.copy()
                mesh_copy.apply_translation(point)
                meshes.append(mesh_copy)
            scene = trimesh.Scene(meshes)
            scene.export(output_path, file_type="gltf")
            
            # GLTF 파일 수정 (accessor min/max, bufferView target, buffer length)
            print("  GLTF 메타데이터 수정 중...")
            fix_gltf_accessors(output_path)
            print(f"GLTF 파일 저장 완료 (포인트 클라우드): {output_path}")
        else:
            raise ValueError(f"지원되지 않는 메쉬 타입: {type(mesh)}")

def generate_text_prompt(category: str) -> str:
    """카테고리로부터 텍스트 프롬프트를 생성합니다.
    
    Args:
        category: 이미지 카테고리
    
    Returns:
        3D 생성에 사용할 텍스트 프롬프트
    """
    # 간단한 프롬프트 생성 (실제 구현에 맞게 수정 가능)
    prompt = f"a 3D model of {category}"
    return prompt

def main():
    """메인 추론 함수"""
    # 디렉토리 생성
    create_directories()
    
    # 디바이스 설정
    device = get_device()
    
    # 베이스 모델 로드
    print("TripodSR 베이스 모델 로드 중...")
    base_model = load_tripodsr_model(device)
    
    # LoRA 가중치 로드 및 병합
    lora_path = DEFAULT_LORA_PATH
    if os.path.exists(lora_path):
        model = load_lora_weights(base_model, lora_path, device)
    else:
        print(f"Warning: LoRA 가중치 파일을 찾을 수 없습니다: {lora_path}")
        print("베이스 모델만 사용합니다.")
        model = base_model
    
    # 이미지-카테고리 매핑 로드
    category_map_path = "data/image_category_map.json"
    if not os.path.exists(category_map_path):
        print(f"Error: 카테고리 맵 파일을 찾을 수 없습니다: {category_map_path}")
        return
    
    print(f"카테고리 맵 로드 중: {category_map_path}")
    category_map = load_image_category_map(category_map_path)
    print(f"로드된 이미지-카테고리 매핑: {len(category_map)}개")
    
    # 이미지 디렉토리에서 모든 이미지 로드 (모든 확장자 지원)
    # 배경 제거된 이미지가 있으면 우선 사용
    image_dir = Path("data/raw_images")
    no_bg_dir = image_dir / "no_background"
    
    # 배경 제거된 이미지가 있으면 사용, 없으면 원본 사용
    if no_bg_dir.exists() and any(no_bg_dir.glob("*_no_bg.png")):
        print(f"배경 제거된 이미지 사용: {no_bg_dir}")
        image_paths = sorted(no_bg_dir.glob("*_no_bg.png"))
    else:
        image_paths = []
        image_paths.extend(image_dir.glob("*.jpg"))
        image_paths.extend(image_dir.glob("*.JPG"))
        image_paths.extend(image_dir.glob("*.jpeg"))
        image_paths.extend(image_dir.glob("*.JPEG"))
        image_paths.extend(image_dir.glob("*.png"))
        image_paths.extend(image_dir.glob("*.PNG"))
        image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"Error: {image_dir}에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"처리할 이미지 수: {len(image_paths)}개")
    
    # 각 이미지에 대해 3D 생성 수행
    for idx, image_path in enumerate(image_paths):
        image_name = image_path.name
        print(f"\n[{idx + 1}/{len(image_paths)}] 처리 중: {image_name}")
        
        # 카테고리 확인 (배경 제거된 이미지의 경우 원본 이름으로 매칭)
        # 예: "my_mug_1_no_bg.png" -> "my_mug_1.jpeg"
        original_name = image_name.replace("_no_bg.png", "").replace("_no_bg.PNG", "")
        
        # 원본 확장자 찾기 (jpeg, jpg, png 등)
        matched_name = None
        for ext in [".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG"]:
            candidate_name = original_name + ext
            if candidate_name in category_map:
                matched_name = candidate_name
                break
        
        # 확장자 없이도 시도 (예: "my_mug_1" -> "my_mug_1.jpeg")
        if matched_name is None:
            for ext in [".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG"]:
                if original_name + ext in category_map:
                    matched_name = original_name + ext
                    break
        
        if matched_name is None:
            print(f"Warning: {image_name}에 대한 카테고리를 찾을 수 없습니다. 건너뜁니다.")
            print(f"  시도한 이름: {original_name} (확장자 포함)")
            continue
        
        original_name = matched_name
        
        category_info = category_map[original_name]
        category = category_info["category"]
        confidence = category_info["confidence"]
        
        print(f"  카테고리: {category} (신뢰도: {confidence:.4f})")
        
        # 이미지 로드 (PIL Image로 직접 로드, 투명 배경 처리)
        image = Image.open(image_path)
        # RGBA를 RGB로 변환 (투명 배경을 흰색으로)
        if image.mode == 'RGBA':
            # Alpha 채널 추출
            alpha = image.split()[3]
            
            # Alpha 채널을 numpy 배열로 변환하여 더 정확한 마스킹
            alpha_array = np.array(alpha)
            
            # 반투명 픽셀도 완전히 투명하게 처리 (threshold 적용)
            # Alpha 값이 128 미만이면 완전히 투명으로 처리
            alpha_threshold = 128
            alpha_array = np.where(alpha_array < alpha_threshold, 0, 255)
            alpha_processed = Image.fromarray(alpha_array.astype(np.uint8))
            
            # RGB 이미지 추출
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=alpha_processed)  # 처리된 alpha 채널을 마스크로 사용
            
            image = rgb_image
            print(f"  배경 제거 이미지 처리 완료 (alpha threshold: {alpha_threshold})")
        else:
            image = image.convert("RGB")
        
        # 텍스트 프롬프트 생성 (현재는 사용하지 않지만 나중을 위해 유지)
        text_prompt = generate_text_prompt(category)
        print(f"  카테고리 프롬프트: {text_prompt}")
        
        # 3D 생성
        print("  3D 모델 생성 중...")
        with torch.no_grad():
            try:
                # TripoSR의 forward로 scene_codes 생성
                scene_codes = model(image, device=str(device))  # type: ignore
                
                # scene_codes로부터 메쉬 추출
                meshes = model.extract_mesh(  # type: ignore
                    scene_codes, 
                    has_vertex_color=True, 
                    resolution=256,
                    threshold=25.0
                )
                
                if not meshes:
                    print(f"  Warning: 메쉬 생성 실패")
                    continue
                
                mesh_result = meshes[0]  # 첫 번째 메쉬 사용
                
            except Exception as e:
                print(f"  Error: 3D 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # GLTF 형식으로 저장
        output_path = f"outputs/gltf_models/{image_path.stem}.gltf"
        print(f"  GLTF 파일로 저장 중: {output_path}")
        
        try:
            mesh_to_gltf(mesh_result, output_path)
        except Exception as e:
            print(f"  Error: GLTF 저장 중 오류 발생: {e}")
            continue
    
    print("\n모든 추론 작업 완료!")

if __name__ == "__main__":
    main()
