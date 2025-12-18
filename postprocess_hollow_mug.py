# 컵 메쉬 후처리: 속이 빈 컵 만들기
# 판막 제거하고 내부 파내기

import os
import argparse
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import trimesh

# manifold3d 또는 pyembree (선택적)
try:
    import manifold3d  # type: ignore
    MANIFOLD3D_AVAILABLE = True
except ImportError:
    MANIFOLD3D_AVAILABLE = False

try:
    import pyembree  # type: ignore
    PYEMBREE_AVAILABLE = True
except ImportError:
    PYEMBREE_AVAILABLE = False


# 메쉬 파일 불러오기
def load_mesh_file(file_path: Path) -> Optional[trimesh.Trimesh]:
    try:
        scene = trimesh.load(str(file_path))
        
        # Scene 객체인 경우 모든 메쉬 합치기
        if isinstance(scene, trimesh.Scene):
            meshes = []
            for node_name in scene.graph.nodes_geometry:
                geometry = scene.geometry[node_name]
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            
            if not meshes:
                return None
            
            if len(meshes) == 1:
                return meshes[0]
            else:
                combined = trimesh.util.concatenate(meshes)
                return combined
        
        # 이미 Trimesh 객체인 경우
        if isinstance(scene, trimesh.Trimesh):
            return scene
        
        # 리스트인 경우
        if isinstance(scene, list):
            if len(scene) == 0:
                return None
            if len(scene) == 1:
                return scene[0]
            combined = trimesh.util.concatenate([m for m in scene if isinstance(m, trimesh.Trimesh)])
            return combined
        
        return None
    except Exception as e:
        print(f"  ✗ 메쉬 로드 실패: {e}")
        return None


# 작은 컴포넌트 제거
def remove_small_components(mesh: trimesh.Trimesh, min_faces: int = 500) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    
    components = mesh.split(only_watertight=False)
    if not components or len(components) == 1:
        return mesh
    
    # 면 개수가 충분한 컴포넌트만 유지
    keep_components = [c for c in components if len(c.faces) >= min_faces]
    
    if not keep_components:
        print(f"  ⚠ Warning: 모든 컴포넌트가 제거됨, 원본 반환")
        return mesh
    
    if len(keep_components) == 1:
        return keep_components[0]
    
    combined = trimesh.util.concatenate(keep_components)
    removed_count = len(components) - len(keep_components)
    print(f"  ✓ 작은 컴포넌트 제거: {removed_count}개 제거")
    
    return combined


# 판막 제거
def remove_plane_artifacts(mesh: trimesh.Trimesh, extent_ratio_threshold: float = 0.02) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    
    components = mesh.split(only_watertight=False)
    if not components or len(components) == 1:
        return mesh
    
    keep_components = []
    removed_count = 0
    
    for comp in components:
        if len(comp.faces) == 0:
            continue
        
        bounds = comp.bounds
        extents = bounds[1] - bounds[0]
        max_extent = np.max(extents)
        
        if max_extent < 1e-10:
            removed_count += len(comp.faces)
            continue
        
        min_extent = np.min(extents)
        extent_ratio = min_extent / max_extent
        
        # 매우 얇고 큰 컴포넌트 제거
        if extent_ratio < extent_ratio_threshold:
            removed_count += len(comp.faces)
        else:
            keep_components.append(comp)
    
    if not keep_components:
        print(f"  ⚠ Warning: 모든 컴포넌트가 제거됨, 원본 반환")
        return mesh
    
    if len(keep_components) == 1:
        result = keep_components[0]
        if removed_count > 0:
            print(f"  ✓ 판막 제거: {removed_count}개 면 제거")
        return result
    
    result = trimesh.util.concatenate(keep_components)
    if removed_count > 0:
        print(f"  ✓ 판막 제거: {removed_count}개 면 제거 ({len(components)-len(keep_components)}개 컴포넌트)")
    
    return result


# 가장 긴 축 찾기 (위쪽 방향)
def detect_up_axis(mesh: trimesh.Trimesh) -> int:
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    up_axis = int(np.argmax(extents))
    return up_axis


# 컵 내부 파내기
def create_hollow_mug(
    mesh: trimesh.Trimesh,
    rim_percentile: float = 98.0,
    wall_thickness_ratio: float = 0.08,
    depth_ratio: float = 0.65
) -> Optional[trimesh.Trimesh]:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    
    vertices = np.asarray(mesh.vertices)
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    
    # up 축 감지
    up_axis = detect_up_axis(mesh)
    print(f"  - Up 축 감지: {'XYZ'[up_axis]}축")
    
    # up 축 좌표 추출
    up_coords = vertices[:, up_axis]
    min_z = bounds[0][up_axis]
    max_z = bounds[1][up_axis]
    height = extents[up_axis]
    
    # 상단 림 영역 추정 (rim_percentile 이상)
    rim_threshold = np.percentile(up_coords, rim_percentile)
    rim_mask = up_coords >= rim_threshold
    rim_vertices = vertices[rim_mask]
    
    if len(rim_vertices) < 3:
        print(f"  ⚠ Warning: 림 영역 정점 부족, hollow 스킵")
        return None
    
    # XY 평면에서 중심 추정 (up 축이 아닌 나머지 두 축)
    other_axes = [i for i in range(3) if i != up_axis]
    rim_xy = rim_vertices[:, other_axes]
    
    # 중심 추정 (중앙값 사용)
    center_xy = np.median(rim_xy, axis=0)
    
    # 외부 반경 추정 (중심으로부터 거리의 퍼센타일)
    distances = np.linalg.norm(rim_xy - center_xy, axis=1)
    outer_radius = np.percentile(distances, 95)
    
    if outer_radius < 1e-6:
        print(f"  ⚠ Warning: 외부 반경이 너무 작음, hollow 스킵")
        return None
    
    inner_radius = outer_radius * (1 - wall_thickness_ratio)
    depth = height * depth_ratio
    
    print(f"  - 외부 반경: {outer_radius:.4f}, 내부 반경: {inner_radius:.4f}, 깊이: {depth:.4f}")
    
    # 파낼 깊이 계산 (상단에서 시작)
    carve_start = rim_threshold
    carve_end = carve_start - depth
    
    # Cylinder 생성 (up 축 방향)
    # trimesh의 cylinder는 z축 방향이므로, up 축에 맞게 회전/변환 필요
    cylinder_height = depth
    cylinder_radius = inner_radius
    
    # 기본 z축 방향 cylinder 생성
    cylinder = trimesh.creation.cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        sections=32
    )
    
    # up 축에 맞게 변환
    if up_axis == 0:  # x축이 up
        # z축 -> x축 회전
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        cylinder.apply_transform(rotation)
    elif up_axis == 1:  # y축이 up
        # z축 -> y축 회전
        rotation = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        cylinder.apply_transform(rotation)
    # up_axis == 2 (z축)이면 변환 불필요
    
    # 중심 위치로 이동
    center_3d = np.zeros(3)
    center_3d[other_axes[0]] = center_xy[0]
    center_3d[other_axes[1]] = center_xy[1]
    center_3d[up_axis] = carve_start - depth / 2
    
    translation = trimesh.transformations.translation_matrix(center_3d)
    cylinder.apply_transform(translation)
    
    # Boolean difference 수행
    try:
        # manifold3d 사용 가능하면 사용
        if MANIFOLD3D_AVAILABLE:
            print(f"  - manifold3d로 boolean difference 수행 중...")
            # trimesh의 boolean은 manifold3d를 자동으로 사용
            result = mesh.difference(cylinder)
            if isinstance(result, trimesh.Trimesh) and len(result.vertices) > 0:
                print(f"  ✓ Hollow 성공 (manifold3d)")
                return result
        
        # 일반 boolean difference 시도
        print(f"  - trimesh boolean difference 수행 중...")
        result = mesh.difference(cylinder)
        
        if isinstance(result, trimesh.Trimesh) and len(result.vertices) > 0:
            print(f"  ✓ Hollow 성공")
            return result
        else:
            print(f"  ⚠ Warning: boolean difference 결과가 비어있음")
            return None
            
    except Exception as e:
        print(f"  ⚠ Warning: boolean difference 실패: {e}")
        
        # Fallback: voxelization 기반 carving 시도
        try:
            print(f"  - Voxelization 기반 carving 시도 중...")
            # 간단한 fallback: 내부 정점 제거 (근사치)
            # 실제로는 복잡하므로 경고만 출력
            print(f"  ⚠ Fallback carving 미구현, hollow 스킵")
            return None
        except Exception as e2:
            print(f"  ⚠ Fallback 실패: {e2}")
            return None


# 메쉬 정리
def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    if len(vertices) == 0 or len(faces) == 0:
        return mesh
    
    v_count = len(vertices)
    
    # vertex colors 추출
    colors: Optional[np.ndarray] = None
    visual_obj = getattr(mesh, "visual", None)
    if visual_obj is not None:
        vc = getattr(visual_obj, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if len(vc_arr) == v_count:
                colors = vc_arr
    
    # OOB face 제거
    valid_face_mask = np.all((faces >= 0) & (faces < v_count), axis=1)
    faces = faces[valid_face_mask]
    
    if len(faces) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    # 사용되는 정점만 남기기
    used_vertices = np.unique(faces.reshape(-1))
    used_vertices = used_vertices[(used_vertices >= 0) & (used_vertices < v_count)]
    
    if len(used_vertices) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    new_vertices = vertices[used_vertices]
    
    # 인덱스 재매핑
    index_map = np.full(v_count, -1, dtype=np.int64)
    index_map[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    
    new_faces = index_map[faces]
    valid_face_mask2 = np.all(new_faces >= 0, axis=1)
    new_faces = new_faces[valid_face_mask2]
    
    if len(new_faces) == 0:
        return trimesh.Trimesh(vertices=new_vertices, faces=np.zeros((0, 3), dtype=np.int64), process=False)
    
    # Degenerate face 제거
    degenerate_mask = np.ones(len(new_faces), dtype=bool)
    for i, face in enumerate(new_faces):
        if len(np.unique(face)) < 3:
            degenerate_mask[i] = False
            continue
        
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
    
    # vertex colors 재매핑
    new_colors: Optional[np.ndarray] = None
    if colors is not None:
        new_colors = colors[used_vertices]
    
    # 새 메쉬 생성
    cleaned = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
    
    # vertex colors 설정
    if new_colors is not None:
        vis2 = getattr(cleaned, "visual", None)
        if vis2 is not None:
            try:
                setattr(vis2, "vertex_colors", new_colors)
            except Exception:
                pass
    
    # trimesh 메서드 호출 (가능하면)
    if hasattr(cleaned, "remove_degenerate_faces"):
        try:
            getattr(cleaned, "remove_degenerate_faces")()
        except Exception:
            pass
    
    if hasattr(cleaned, "merge_vertices"):
        try:
            getattr(cleaned, "merge_vertices")()
        except Exception:
            pass
    
    return cleaned


# 메쉬 파일 처리
def process_mesh_file(
    input_path: Path,
    output_path: Path,
    hollow: bool = True,
    plane_remove: bool = True,
    rim_percentile: float = 98.0,
    wall_thickness_ratio: float = 0.08,
    depth_ratio: float = 0.65,
    min_component_faces: int = 500
) -> bool:
    print(f"\n처리 중: {input_path.name} -> {output_path.name}")
    
    # (A) 로드
    mesh = load_mesh_file(input_path)
    if mesh is None:
        print(f"  ✗ 메쉬 로드 실패")
        return False
    
    initial_vertices = len(mesh.vertices)
    initial_faces = len(mesh.faces)
    print(f"  - 입력: {initial_vertices}개 정점, {initial_faces}개 면")
    
    # (C) 작은 컴포넌트 제거
    mesh = remove_small_components(mesh, min_component_faces)
    
    # (D) 판막 제거 (옵션)
    if plane_remove:
        mesh = remove_plane_artifacts(mesh)
    
    # (E) Hollow (옵션)
    if hollow:
        hollow_result = create_hollow_mug(
            mesh,
            rim_percentile=rim_percentile,
            wall_thickness_ratio=wall_thickness_ratio,
            depth_ratio=depth_ratio
        )
        if hollow_result is not None:
            mesh = hollow_result
        else:
            print(f"  ⚠ Hollow 실패, 원본 메쉬 사용")
    
    # (F) 후정리
    mesh = clean_mesh(mesh)
    
    final_vertices = len(mesh.vertices)
    final_faces = len(mesh.faces)
    print(f"  - 출력: {final_vertices}개 정점, {final_faces}개 면")
    
    # GLB 내보내기
    try:
        scene = trimesh.Scene([mesh])
        scene.export(str(output_path), file_type="glb")
        print(f"  ✅ 저장 완료: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ GLB 내보내기 실패: {e}")
        return False


# 디렉토리 전체 처리
def process_directory(
    input_dir: Path,
    output_dir: Path,
    **kwargs
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_paths = []
    for ext in [".glb", ".GLB", ".gltf", ".GLTF"]:
        mesh_paths.extend(input_dir.glob(f"*{ext}"))
    
    if not mesh_paths:
        print(f"Warning: {input_dir}에서 메쉬 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 메쉬 파일: {len(mesh_paths)}개")
    print(f"출력 디렉토리: {output_dir}\n")
    
    success_count = 0
    for mesh_path in sorted(mesh_paths):
        output_path = output_dir / f"{mesh_path.stem}.glb"
        if process_mesh_file(mesh_path, output_path, **kwargs):
            success_count += 1
    
    print(f"\n완료: {success_count}/{len(mesh_paths)}개 처리 성공")


def main():
    parser = argparse.ArgumentParser(description="컵 메쉬 후처리: 속이 빈 컵 생성")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="outputs/glb_models",
        help="입력 메쉬 파일 또는 디렉토리 (기본값: outputs/glb_models)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 파일 또는 디렉토리 (기본값: outputs/clean_models)"
    )
    parser.add_argument(
        "--hollow",
        action="store_true",
        default=True,
        help="Hollow 처리 활성화 (기본값: True)"
    )
    parser.add_argument(
        "--no-hollow",
        action="store_false",
        dest="hollow",
        help="Hollow 처리 비활성화"
    )
    parser.add_argument(
        "--plane-remove",
        action="store_true",
        default=True,
        help="판막 제거 활성화 (기본값: True)"
    )
    parser.add_argument(
        "--no-plane-remove",
        action="store_false",
        dest="plane_remove",
        help="판막 제거 비활성화"
    )
    parser.add_argument(
        "--rim-percentile",
        type=float,
        default=98.0,
        help="상단 림 추정 퍼센타일 (기본값: 98.0)"
    )
    parser.add_argument(
        "--wall-thickness-ratio",
        type=float,
        default=0.08,
        help="벽 두께 비율 (기본값: 0.08)"
    )
    parser.add_argument(
        "--depth-ratio",
        type=float,
        default=0.65,
        help="파낼 깊이 비율 (기본값: 0.65)"
    )
    parser.add_argument(
        "--min-component-faces",
        type=int,
        default=500,
        help="최소 컴포넌트 면 개수 (기본값: 500)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 입력 경로를 찾을 수 없습니다: {input_path}")
        return
    
    if args.output is None:
        output_path = Path("outputs/clean_models")
    else:
        output_path = Path(args.output)
    
    kwargs = {
        "hollow": args.hollow,
        "plane_remove": args.plane_remove,
        "rim_percentile": args.rim_percentile,
        "wall_thickness_ratio": args.wall_thickness_ratio,
        "depth_ratio": args.depth_ratio,
        "min_component_faces": args.min_component_faces,
    }
    
    try:
        if input_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            process_mesh_file(input_path, output_path, **kwargs)
        elif input_path.is_dir():
            process_directory(input_path, output_path, **kwargs)
        else:
            print(f"Error: 잘못된 입력 경로입니다: {input_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

