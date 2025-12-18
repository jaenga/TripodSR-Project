# 메쉬 후처리: 판막 아티팩트 제거
# 1. 가장 큰 컴포넌트만 남기기
# 2. 사용 안 하는 정점 제거
# 3. 잘못된 면 제거
# 4. 얇은 판막 제거

import os
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import trimesh
import argparse


# 가장 큰 컴포넌트만 남기기
def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    if len(mesh.faces) == 0:
        return mesh
    
    # 메쉬를 연결된 컴포넌트로 분리
    components = mesh.split(only_watertight=False)
    
    if not components or len(components) == 1:
        return mesh
    
    # 가장 큰 컴포넌트 찾기 (면 개수 기준)
    largest_component = max(components, key=lambda m: len(m.faces))
    
    print(f"  ✓ Connected Components: {len(components)}개 컴포넌트 발견, 가장 큰 것만 유지 (면: {len(largest_component.faces)}개)")
    
    return largest_component


# 사용 안 하는 정점 제거
def remove_unreferenced_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    if len(mesh.faces) == 0:
        return mesh
    
    # vertex colors 처리
    colors = None
    visual_obj = getattr(mesh, "visual", None)
    if visual_obj is not None:
        vc = getattr(visual_obj, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if len(vc_arr) == len(mesh.vertices):
                colors = vc_arr
    
    # trimesh의 내장 메서드 사용
    remove_unreferenced_vertices_method = getattr(mesh, "remove_unreferenced_vertices", None)
    if remove_unreferenced_vertices_method is not None:
        try:
            remove_unreferenced_vertices_method()
            if colors is not None:
                # 색상도 같이 업데이트
                used_vertices = np.unique(mesh.faces.reshape(-1))
                if len(used_vertices) == len(mesh.vertices):
                    vis2 = getattr(mesh, "visual", None)
                    if vis2 is not None:
                        try:
                            setattr(vis2, "vertex_colors", colors[used_vertices])
                        except Exception:
                            pass
            print(f"  ✓ Remove Unreferenced Vertices: 완료")
            return mesh
        except Exception:
            pass
    
    # 수동으로 처리
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    # 사용되는 정점만 찾기
    used_vertices = np.unique(faces.reshape(-1))
    used_vertices = used_vertices[(used_vertices >= 0) & (used_vertices < len(vertices))]
    
    if len(used_vertices) == len(vertices):
        return mesh
    
    # 새 정점 배열 생성
    new_vertices = vertices[used_vertices]
    
    # 인덱스 재매핑
    index_map = np.full(len(vertices), -1, dtype=np.int64)
    index_map[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    
    new_faces = index_map[faces]
    valid_mask = np.all(new_faces >= 0, axis=1)
    new_faces = new_faces[valid_mask]
    
    # 색상 처리
    new_colors = None
    if colors is not None:
        new_colors = colors[used_vertices]
    
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
    
    if new_colors is not None:
        vis2 = getattr(new_mesh, "visual", None)
        if vis2 is not None:
            try:
                setattr(vis2, "vertex_colors", new_colors)
            except Exception:
                pass
    
    removed_count = len(vertices) - len(new_vertices)
    print(f"  ✓ Remove Unreferenced Vertices: {removed_count}개 정점 제거")
    
    return new_mesh


# 잘못된 면 제거
def remove_degenerate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    if len(mesh.faces) == 0:
        return mesh
    
    initial_face_count = len(mesh.faces)
    
    # trimesh의 내장 메서드 사용
    remove_degenerate_faces_method = getattr(mesh, "remove_degenerate_faces", None)
    if remove_degenerate_faces_method is not None:
        try:
            remove_degenerate_faces_method()
            removed_count = initial_face_count - len(mesh.faces)
            if removed_count > 0:
                print(f"  ✓ Degenerate Face 제거: {removed_count}개 면 제거")
            else:
                print(f"  ✓ Degenerate Face 제거: 퇴화된 면 없음")
            return mesh
        except Exception:
            pass
    
    # 수동으로 처리: 면적이 0이거나 매우 작은 면 제거
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    # 각 면의 면적 계산
    face_areas = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        # 삼각형 면적: ||(v1-v0) × (v2-v0)|| / 2
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross)
        face_areas.append(area)
    
    face_areas = np.array(face_areas)
    
    # 면적이 매우 작은 면 제거 (1e-10 이하)
    min_area = 1e-10
    valid_mask = face_areas > min_area
    
    # 중복 정점이 있는 면도 제거
    for i, face in enumerate(faces):
        if len(np.unique(face)) < 3:
            valid_mask[i] = False
    
    if np.all(valid_mask):
        print(f"  ✓ Degenerate Face 제거: 퇴화된 면 없음")
        return mesh
    
    keep_faces = faces[valid_mask]
    
    # 사용되는 정점만 남기기
    used_vertices = np.unique(keep_faces.reshape(-1))
    new_vertices = vertices[used_vertices]
    
    # 인덱스 재매핑
    index_map = np.full(len(vertices), -1, dtype=np.int64)
    index_map[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    
    new_faces = index_map[keep_faces]
    valid_mask2 = np.all(new_faces >= 0, axis=1)
    new_faces = new_faces[valid_mask2]
    
    # vertex colors 처리
    colors = None
    visual_obj = getattr(mesh, "visual", None)
    if visual_obj is not None:
        vc = getattr(visual_obj, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if len(vc_arr) == len(vertices):
                colors = vc_arr[used_vertices]
    
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
    
    if colors is not None:
        vis2 = getattr(new_mesh, "visual", None)
        if vis2 is not None:
            try:
                setattr(vis2, "vertex_colors", colors)
            except Exception:
                pass
    
    removed_count = initial_face_count - len(new_faces)
    print(f"  ✓ Degenerate Face 제거: {removed_count}개 면 제거")
    
    return new_mesh


# 얇은 판막 제거
def remove_thin_planes(
    mesh: trimesh.Trimesh,
    extent_ratio_threshold: float = 0.02,
    min_faces: int = 10
) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    if len(mesh.faces) == 0:
        return mesh
    
    # 메쉬를 연결된 컴포넌트로 분리
    components = mesh.split(only_watertight=False)
    
    if not components or len(components) == 1:
        # 컴포넌트가 하나뿐이면 판단 불가
        print(f"  ✓ Thin Plane 제거: 컴포넌트가 1개뿐이므로 판단 불가")
        return mesh
    
    # 판막이 아닌 컴포넌트만 저장
    keep_components = []
    removed_count = 0
    
    for comp in components:
        if len(comp.faces) == 0:
            continue
        
        # AABB extents 계산
        bounds = comp.bounds
        extents = bounds[1] - bounds[0]  # [width, height, depth]
        
        # 0으로 나누기 방지
        max_extent = np.max(extents)
        if max_extent < 1e-10:
            # 매우 작은 컴포넌트는 제거
            removed_count += len(comp.faces)
            continue
        
        min_extent = np.min(extents)
        extent_ratio = min_extent / max_extent
        
        # 판막 판단: extent_ratio가 임계값 이하이고 면 개수가 일정 이상
        is_thin_plane = (extent_ratio < extent_ratio_threshold and 
                        len(comp.faces) >= min_faces)
        
        if is_thin_plane:
            removed_count += len(comp.faces)
            print(f"    - 판막 제거: {len(comp.faces)}개 면, extent_ratio={extent_ratio:.4f}, extents={extents}")
        else:
            keep_components.append(comp)
    
    # 남은 컴포넌트가 없으면 원본 반환
    if not keep_components:
        print(f"  ⚠ Warning: 모든 컴포넌트가 제거됨, 원본 메쉬 반환")
        return mesh
    
    # 남은 컴포넌트가 하나면 그대로 반환
    if len(keep_components) == 1:
        result = keep_components[0]
        print(f"  ✓ Thin Plane 제거: {removed_count}개 면 제거 ({len(components)-1}개 컴포넌트 제거)")
        return result
    
    # 여러 컴포넌트를 합치기
    result = trimesh.util.concatenate(keep_components)
    
    print(f"  ✓ Thin Plane 제거: {removed_count}개 면 제거 ({len(components)-len(keep_components)}개 컴포넌트 제거)")
    
    return result


# 메쉬 후처리 전체 파이프라인
def postprocess_mesh(
    mesh: trimesh.Trimesh,
    use_thin_plane_removal: bool = True,
    extent_ratio_threshold: float = 0.02,
    min_faces: int = 10
) -> trimesh.Trimesh:
    result = mesh
    
    # 1. Connected Components: 가장 큰 컴포넌트만 남기기
    result = keep_largest_component(result)
    
    # 2. Remove Unreferenced Vertices
    result = remove_unreferenced_vertices(result)
    
    # 3. Degenerate Face 제거
    result = remove_degenerate_faces(result)
    
    # 4. Thin Plane 제거 (선택적)
    if use_thin_plane_removal:
        result = remove_thin_planes(result, extent_ratio_threshold, min_faces)
    
    return result


# GLTF 파일 불러오기
def load_gltf_with_bin(file_path: Path) -> Optional[trimesh.Trimesh]:
    try:
        # trimesh가 자동으로 .bin 파일을 찾아서 로드함
        scene = trimesh.load(str(file_path))
        
        # Scene 객체인 경우 첫 번째 메쉬 추출
        if isinstance(scene, trimesh.Scene):
            # 모든 메쉬를 하나로 합치기
            meshes = []
            for node_name in scene.graph.nodes_geometry:
                geometry = scene.geometry[node_name]
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            
            if meshes:
                if len(meshes) == 1:
                    return meshes[0]
                else:
                    # 여러 메쉬를 합치기
                    combined = trimesh.util.concatenate(meshes)
                    return combined
            else:
                return None
        
        # 이미 Trimesh 객체인 경우
        if isinstance(scene, trimesh.Trimesh):
            return scene
        
        # 리스트인 경우
        if isinstance(scene, list):
            if len(scene) == 0:
                return None
            if len(scene) == 1:
                return scene[0]
            # 여러 메쉬를 합치기
            combined = trimesh.util.concatenate([m for m in scene if isinstance(m, trimesh.Trimesh)])
            return combined
        
        return None
        
    except Exception as e:
        print(f"  ✗ GLTF 로드 실패: {e}")
        return None


# 메쉬 파일 처리
def process_mesh_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    use_thin_plane_removal: bool = True,
    extent_ratio_threshold: float = 0.02,
    min_faces: int = 10
) -> Optional[str]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 메쉬 파일을 찾을 수 없습니다: {input_path}")
    
    # 출력 경로 설정
    if output_path is None:
        output_dir = Path("outputs/clean_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        # 입력 파일명 사용, 확장자는 .glb로 변경
        output_path = output_dir / f"{input_path.stem}.glb"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n후처리 중: {input_path.name} -> {output_path.name}")
    
    # 메쉬 로드 (GLTF+bin 지원)
    if input_path.suffix.lower() in ['.gltf', '.glb']:
        mesh = load_gltf_with_bin(input_path)
    else:
        mesh = trimesh.load(str(input_path))
    
    if mesh is None:
        raise ValueError(f"메쉬를 로드할 수 없습니다: {input_path}")
    
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, list):
            if len(mesh) == 0:
                raise ValueError(f"메쉬가 비어있습니다: {input_path}")
            mesh = mesh[0]
        else:
            raise ValueError(f"지원하지 않는 메쉬 타입: {type(mesh)}")
    
    print(f"  - 입력 메쉬: {len(mesh.vertices)}개 정점, {len(mesh.faces)}개 면")
    
    # 후처리
    processed_mesh = postprocess_mesh(mesh, use_thin_plane_removal, extent_ratio_threshold, min_faces)
    
    print(f"  - 출력 메쉬: {len(processed_mesh.vertices)}개 정점, {len(processed_mesh.faces)}개 면")
    
    # 결과 저장 (.glb 형식)
    processed_mesh.export(str(output_path), file_type="glb")
    print(f"✓ 후처리 완료: {output_path}")
    
    return str(output_path)


# 디렉토리 전체 처리
def process_directory(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    use_thin_plane_removal: bool = True,
    extent_ratio_threshold: float = 0.02,
    min_faces: int = 10
):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
    
    if output_dir is None:
        output_dir = Path("outputs/clean_models")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 지원하는 메쉬 확장자
    mesh_extensions = ['.glb', '.gltf', '.obj', '.ply', '.stl', '.off']
    
    # 메쉬 파일 찾기
    mesh_paths = []
    for ext in mesh_extensions:
        mesh_paths.extend(input_dir.glob(f"*{ext}"))
        mesh_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not mesh_paths:
        print(f"Warning: {input_dir}에서 메쉬 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 메쉬 파일: {len(mesh_paths)}개")
    print(f"출력 디렉토리: {output_dir}\n")
    
    success_count = 0
    for idx, mesh_path in enumerate(sorted(mesh_paths), 1):
        try:
            # 출력 파일명: 입력 파일명 사용, 확장자는 .glb로 변경
            output_path = output_dir / f"{mesh_path.stem}.glb"
            process_mesh_file(mesh_path, output_path, use_thin_plane_removal, extent_ratio_threshold, min_faces)
            success_count += 1
        except Exception as e:
            print(f"✗ 오류 발생 ({mesh_path.name}): {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n완료: {success_count}/{len(mesh_paths)}개 메쉬 파일 처리 완료")


def main():
    parser = argparse.ArgumentParser(description="메쉬 후처리 도구 - 판막 아티팩트 제거")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="outputs/gltf_models",
        help="입력 메쉬 파일 또는 디렉토리 경로 (기본값: outputs/gltf_models)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 파일 또는 디렉토리 경로 (지정하지 않으면 outputs/clean_models)"
    )
    parser.add_argument(
        "--no-thin-plane",
        action="store_true",
        help="Thin Plane 제거 비활성화"
    )
    parser.add_argument(
        "--extent-ratio-threshold",
        type=float,
        default=0.02,
        help="min_extent/max_extent 비율 임계값 (기본값: 0.02)"
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=10,
        help="판막으로 판단할 최소 면 개수 (기본값: 10)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: 입력 경로를 찾을 수 없습니다: {input_path}")
        return
    
    kwargs = {
        "use_thin_plane_removal": not args.no_thin_plane,
        "extent_ratio_threshold": args.extent_ratio_threshold,
        "min_faces": args.min_faces,
    }
    
    try:
        if input_path.is_file():
            process_mesh_file(input_path, args.output, **kwargs)
        elif input_path.is_dir():
            process_directory(input_path, args.output, **kwargs)
        else:
            print(f"Error: 잘못된 입력 경로입니다: {input_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
