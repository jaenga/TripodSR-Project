# 메쉬 품질 평가: 정점/면 수, 판막 제거율 등 계산

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import trimesh


# 메쉬 통계 계산
def calculate_mesh_statistics(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    if mesh is None or len(mesh.vertices) == 0:
        return {
            "num_vertices": 0,
            "num_faces": 0,
            "num_components": 0,
            "volume": 0.0,
            "surface_area": 0.0,
            "bounding_box_volume": 0.0,
            "compactness": 0.0,
            "extent_ratios": [0.0, 0.0, 0.0],
            "min_extent_ratio": 0.0,
        }
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    # 기본 통계
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # 컴포넌트 분석
    components = mesh.split(only_watertight=False)
    num_components = len(components) if components else 0
    
    # 부피 및 표면적
    try:
        volume = mesh.volume if mesh.is_watertight else 0.0
    except Exception:
        volume = 0.0
    
    try:
        surface_area = mesh.area
    except Exception:
        surface_area = 0.0
    
    # 경계 박스
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    bbox_volume = np.prod(extents)
    
    # Compactness (부피 대비 표면적 비율, 낮을수록 좋음)
    compactness = surface_area / (bbox_volume ** (2/3)) if bbox_volume > 0 else 0.0
    
    # Extent ratios (각 축의 비율)
    max_extent = np.max(extents)
    extent_ratios = extents / max_extent if max_extent > 0 else extents
    min_extent_ratio = np.min(extent_ratios)
    
    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "num_components": num_components,
        "volume": float(volume),
        "surface_area": float(surface_area),
        "bounding_box_volume": float(bbox_volume),
        "compactness": float(compactness),
        "extent_ratios": extent_ratios.tolist(),
        "min_extent_ratio": float(min_extent_ratio),
    }


# 판막 분석
def analyze_thin_planes(mesh: trimesh.Trimesh, extent_ratio_threshold: float = 0.02) -> Dict[str, Any]:
    if mesh is None or len(mesh.faces) == 0:
        return {
            "total_components": 0,
            "thin_plane_components": 0,
            "thin_plane_faces": 0,
            "thin_plane_ratio": 0.0,
            "details": [],
        }
    
    components = mesh.split(only_watertight=False)
    if not components:
        return {
            "total_components": 0,
            "thin_plane_components": 0,
            "thin_plane_faces": 0,
            "thin_plane_ratio": 0.0,
            "details": [],
        }
    
    total_faces = len(mesh.faces)
    thin_plane_faces = 0
    thin_plane_components = 0
    details = []
    
    for i, comp in enumerate(components):
        if len(comp.faces) == 0:
            continue
        
        bounds = comp.bounds
        extents = bounds[1] - bounds[0]
        max_extent = np.max(extents)
        
        if max_extent < 1e-10:
            continue
        
        min_extent = np.min(extents)
        extent_ratio = min_extent / max_extent
        
        is_thin_plane = extent_ratio < extent_ratio_threshold
        
        comp_info = {
            "component_id": i,
            "num_faces": len(comp.faces),
            "extent_ratio": float(extent_ratio),
            "extents": extents.tolist(),
            "is_thin_plane": bool(is_thin_plane),
        }
        details.append(comp_info)
        
        if is_thin_plane:
            thin_plane_components += 1
            thin_plane_faces += len(comp.faces)
    
    thin_plane_ratio = thin_plane_faces / total_faces if total_faces > 0 else 0.0
    
    return {
        "total_components": len(components),
        "thin_plane_components": thin_plane_components,
        "thin_plane_faces": thin_plane_faces,
        "thin_plane_ratio": float(thin_plane_ratio),
        "details": details,
    }


# 전처리 전후 비교
def compare_meshes(
    before_path: Path,
    after_path: Path,
    extent_ratio_threshold: float = 0.02
) -> Dict[str, Any]:
    print(f"\n비교 분석: {before_path.name} vs {after_path.name}")
    
    # 메쉬 로드
    before_scene = trimesh.load(str(before_path))
    after_scene = trimesh.load(str(after_path))
    
    # Scene 처리
    if isinstance(before_scene, trimesh.Scene):
        before_meshes = [g for g in before_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        before_mesh = trimesh.util.concatenate(before_meshes) if len(before_meshes) > 1 else (before_meshes[0] if before_meshes else None)
    elif isinstance(before_scene, trimesh.Trimesh):
        before_mesh = before_scene
    elif isinstance(before_scene, list):
        before_mesh = trimesh.util.concatenate([m for m in before_scene if isinstance(m, trimesh.Trimesh)])
    else:
        before_mesh = None
    
    if isinstance(after_scene, trimesh.Scene):
        after_meshes = [g for g in after_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        after_mesh = trimesh.util.concatenate(after_meshes) if len(after_meshes) > 1 else (after_meshes[0] if after_meshes else None)
    elif isinstance(after_scene, trimesh.Trimesh):
        after_mesh = after_scene
    elif isinstance(after_scene, list):
        after_mesh = trimesh.util.concatenate([m for m in after_scene if isinstance(m, trimesh.Trimesh)])
    else:
        after_mesh = None
    
    if before_mesh is None or after_mesh is None:
        print("  ✗ 메쉬 로드 실패")
        return {}
    
    # 통계 계산
    before_stats = calculate_mesh_statistics(before_mesh)
    after_stats = calculate_mesh_statistics(after_mesh)
    
    # 판막 분석
    before_planes = analyze_thin_planes(before_mesh, extent_ratio_threshold)
    after_planes = analyze_thin_planes(after_mesh, extent_ratio_threshold)
    
    # 변화량 계산
    before_vertices = int(before_stats["num_vertices"])
    after_vertices = int(after_stats["num_vertices"])
    before_faces = int(before_stats["num_faces"])
    after_faces = int(after_stats["num_faces"])
    before_components = int(before_stats["num_components"])
    after_components = int(after_stats["num_components"])
    
    vertex_reduction = before_vertices - after_vertices
    vertex_reduction_ratio = vertex_reduction / before_vertices if before_vertices > 0 else 0.0
    
    face_reduction = before_faces - after_faces
    face_reduction_ratio = face_reduction / before_faces if before_faces > 0 else 0.0
    
    before_plane_faces = int(before_planes["thin_plane_faces"])
    after_plane_faces = int(after_planes["thin_plane_faces"])
    plane_reduction = before_plane_faces - after_plane_faces
    plane_reduction_ratio = plane_reduction / before_plane_faces if before_plane_faces > 0 else 0.0
    
    component_reduction = before_components - after_components
    
    result = {
        "before": {
            "file": str(before_path),
            "statistics": before_stats,
            "thin_planes": before_planes,
        },
        "after": {
            "file": str(after_path),
            "statistics": after_stats,
            "thin_planes": after_planes,
        },
        "changes": {
            "vertex_reduction": int(vertex_reduction),
            "vertex_reduction_ratio": float(vertex_reduction_ratio),
            "face_reduction": int(face_reduction),
            "face_reduction_ratio": float(face_reduction_ratio),
            "plane_reduction": int(plane_reduction),
            "plane_reduction_ratio": float(plane_reduction_ratio),
            "component_reduction": int(component_reduction),
        },
    }
    
    return result


# 비교 결과 출력
def print_comparison_report(comparison: Dict[str, Any]):
    before = comparison["before"]
    after = comparison["after"]
    changes = comparison["changes"]
    
    print("\n" + "="*60)
    print("메쉬 품질 비교 리포트")
    print("="*60)
    
    print("\n[Before - 전처리 전]")
    print(f"  정점 수: {before['statistics']['num_vertices']:,}")
    print(f"  면 수: {before['statistics']['num_faces']:,}")
    print(f"  컴포넌트 수: {before['statistics']['num_components']}")
    print(f"  판막 컴포넌트: {before['thin_planes']['thin_plane_components']}")
    print(f"  판막 면 수: {before['thin_planes']['thin_plane_faces']:,}")
    print(f"  판막 비율: {before['thin_planes']['thin_plane_ratio']*100:.2f}%")
    print(f"  표면적: {before['statistics']['surface_area']:.4f}")
    print(f"  Compactness: {before['statistics']['compactness']:.4f}")
    
    print("\n[After - 전처리 후]")
    print(f"  정점 수: {after['statistics']['num_vertices']:,}")
    print(f"  면 수: {after['statistics']['num_faces']:,}")
    print(f"  컴포넌트 수: {after['statistics']['num_components']}")
    print(f"  판막 컴포넌트: {after['thin_planes']['thin_plane_components']}")
    print(f"  판막 면 수: {after['thin_planes']['thin_plane_faces']:,}")
    print(f"  판막 비율: {after['thin_planes']['thin_plane_ratio']*100:.2f}%")
    print(f"  표면적: {after['statistics']['surface_area']:.4f}")
    print(f"  Compactness: {after['statistics']['compactness']:.4f}")
    
    print("\n[변화량]")
    print(f"  정점 감소: {changes['vertex_reduction']:,} ({changes['vertex_reduction_ratio']*100:.2f}%)")
    print(f"  면 감소: {changes['face_reduction']:,} ({changes['face_reduction_ratio']*100:.2f}%)")
    print(f"  판막 면 감소: {changes['plane_reduction']:,} ({changes['plane_reduction_ratio']*100:.2f}%)")
    print(f"  컴포넌트 감소: {changes['component_reduction']}")
    
    print("\n" + "="*60)


# 디렉토리 전체 비교
def process_directory(
    before_dir: Path,
    after_dir: Path,
    output_json: Optional[Path] = None,
    extent_ratio_threshold: float = 0.02
):
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)
    
    # GLB 파일 찾기
    before_files = {}
    for ext in [".glb", ".GLB", ".gltf", ".GLTF"]:
        for f in before_dir.glob(f"*{ext}"):
            before_files[f.stem] = f
    
    after_files = {}
    for ext in [".glb", ".GLB", ".gltf", ".GLTF"]:
        for f in after_dir.glob(f"*{ext}"):
            after_files[f.stem] = f
    
    # 매칭된 파일들 비교
    common_stems = set(before_files.keys()) & set(after_files.keys())
    
    if not common_stems:
        print(f"Warning: 매칭되는 파일이 없습니다.")
        return
    
    print(f"발견된 매칭 파일: {len(common_stems)}개\n")
    
    all_results = []
    total_vertex_reduction = 0
    total_face_reduction = 0
    total_plane_reduction = 0
    
    for stem in sorted(common_stems):
        before_path = before_files[stem]
        after_path = after_files[stem]
        
        comparison = compare_meshes(before_path, after_path, extent_ratio_threshold)
        
        if comparison:
            print_comparison_report(comparison)
            all_results.append(comparison)
            
            total_vertex_reduction += comparison["changes"]["vertex_reduction"]
            total_face_reduction += comparison["changes"]["face_reduction"]
            total_plane_reduction += comparison["changes"]["plane_reduction"]
    
    # 전체 요약
    print("\n" + "="*60)
    print("전체 요약")
    print("="*60)
    print(f"비교된 파일 수: {len(all_results)}")
    print(f"총 정점 감소: {total_vertex_reduction:,}")
    print(f"총 면 감소: {total_face_reduction:,}")
    print(f"총 판막 면 감소: {total_plane_reduction:,}")
    print("="*60)
    
    # JSON 저장
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "num_files": len(all_results),
                    "total_vertex_reduction": total_vertex_reduction,
                    "total_face_reduction": total_face_reduction,
                    "total_plane_reduction": total_plane_reduction,
                },
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n결과가 JSON으로 저장되었습니다: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="메쉬 품질 평가 및 Before/After 비교")
    parser.add_argument(
        "before",
        type=str,
        help="전처리 전 메쉬 파일 또는 디렉토리"
    )
    parser.add_argument(
        "after",
        type=str,
        help="전처리 후 메쉬 파일 또는 디렉토리"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="JSON 출력 파일 경로 (선택사항)"
    )
    parser.add_argument(
        "--extent-ratio-threshold",
        type=float,
        default=0.02,
        help="판막 판단 임계값 (기본값: 0.02)"
    )
    
    args = parser.parse_args()
    
    before_path = Path(args.before)
    after_path = Path(args.after)
    
    if not before_path.exists():
        print(f"Error: Before 경로를 찾을 수 없습니다: {before_path}")
        return
    
    if not after_path.exists():
        print(f"Error: After 경로를 찾을 수 없습니다: {after_path}")
        return
    
    output_json = Path(args.output) if args.output else None
    
    if before_path.is_file() and after_path.is_file():
        # 단일 파일 비교
        comparison = compare_meshes(before_path, after_path, args.extent_ratio_threshold)
        if comparison:
            print_comparison_report(comparison)
            if output_json:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(comparison, f, indent=2, ensure_ascii=False)
                print(f"\n결과가 JSON으로 저장되었습니다: {output_json}")
    elif before_path.is_dir() and after_path.is_dir():
        # 디렉토리 비교
        process_directory(before_path, after_path, output_json, args.extent_ratio_threshold)
    else:
        print("Error: Before와 After는 모두 파일이거나 모두 디렉토리여야 합니다.")


if __name__ == "__main__":
    main()

