# pip install torch torchvision
# pip install Pillow
# pip install trimesh
# pip install open3d
# pip install scikit-image  # SSIM 계산용

import os
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
import trimesh
from triposr_backbone import load_tripodsr_model
from inference import (
    load_lora_weights,
    load_image_category_map,
    preprocess_image_for_triposr,
    generate_3d_mesh,
    mesh_to_gltf
)

try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
    from skimage.metrics import mean_squared_error  # type: ignore
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: scikit-image가 설치되지 않았습니다. SSIM 계산을 건너뜁니다.")


def create_directories():
    """필요한 디렉토리 생성"""
    os.makedirs("outputs/eval", exist_ok=True)


def render_mesh_from_viewpoint(
    mesh: trimesh.Trimesh,
    camera_position: np.ndarray,
    look_at: np.ndarray = np.array([0, 0, 0]),
    up: np.ndarray = np.array([0, 1, 0]),
    resolution: Tuple[int, int] = (512, 512),
    fov: float = 60.0
) -> np.ndarray:
    """메쉬를 특정 뷰포인트에서 렌더링합니다.
    
    Args:
        mesh: trimesh.Trimesh 객체
        camera_position: 카메라 위치 [x, y, z]
        look_at: 카메라가 바라보는 지점 [x, y, z]
        up: 카메라의 위 방향 벡터 [x, y, z]
        resolution: 렌더링 해상도 (width, height)
        fov: 시야각 (degrees)
    
    Returns:
        렌더링된 이미지 (numpy array, uint8, RGB)
    """
    try:
        import open3d as o3d
        
        # trimesh를 open3d로 변환
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Vertex colors가 있으면 적용
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = np.asarray(mesh.visual.vertex_colors)
            if colors.max() > 1.0:
                colors = colors / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
        else:
            # 기본 색상
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        o3d_mesh.compute_vertex_normals()
        
        # 메쉬를 원점 중심으로 정규화 (렌더링 전에 수행)
        bbox = o3d_mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        o3d_mesh.translate(-center)
        
        # 렌더러 설정
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=resolution[0], height=resolution[1], visible=False)
        vis.add_geometry(o3d_mesh)
        
        # 카메라 파라미터 설정
        ctr = vis.get_view_control()
        
        # 카메라 위치를 메쉬 크기에 맞게 조정 (정규화 후 bbox 재계산)
        bbox = o3d_mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_extent = max(extent) if max(extent) > 0 else 1.0
        distance = max_extent * 2.0  # 메쉬가 보이도록 충분한 거리
        
        # 카메라 위치 정규화
        camera_dir = camera_position - look_at
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        adjusted_camera_pos = camera_dir * distance
        
        # 뷰 컨트롤 설정 (더 간단한 방법)
        # look_at을 원점으로, camera_position을 사용
        ctr.set_lookat([0, 0, 0])
        ctr.set_front(-adjusted_camera_pos / np.linalg.norm(adjusted_camera_pos))
        ctr.set_up(up / np.linalg.norm(up))
        ctr.set_zoom(0.7)  # 줌 레벨 조정
        
        # 렌더링
        vis.poll_events()
        vis.update_renderer()
        
        # 이미지 캡처
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        # numpy 배열로 변환 (RGB 순서 유지)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        # Open3D는 BGR 순서일 수 있으므로 RGB로 변환
        if len(image_np.shape) == 3:
            image_np = image_np[:, :, ::-1]  # BGR -> RGB
        
        return image_np
        
    except ImportError:
        # open3d가 없으면 trimesh의 간단한 렌더링 사용
        print("Warning: open3d가 설치되지 않았습니다. 간단한 렌더링을 사용합니다.")
        
        # 간단한 placeholder 이미지 (실제 렌더링 대신)
        image = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 128
        return image
    except Exception as e:
        print(f"Warning: 렌더링 중 오류 발생: {e}")
        # Fallback: 간단한 placeholder 이미지
        image = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 128
        return image


def get_camera_viewpoints() -> List[Dict]:
    """평가에 사용할 카메라 뷰포인트 리스트를 반환합니다.
    
    Returns:
        카메라 위치와 look_at 정보를 담은 딕셔너리 리스트
    """
    # 메쉬가 원점에 있다고 가정하고 여러 각도에서 촬영
    viewpoints = [
        {
            "name": "front",
            "camera_position": np.array([0, 0, 3]),
            "look_at": np.array([0, 0, 0]),
            "up": np.array([0, 1, 0])
        },
        {
            "name": "side",
            "camera_position": np.array([3, 0, 0]),
            "look_at": np.array([0, 0, 0]),
            "up": np.array([0, 1, 0])
        },
        {
            "name": "top",
            "camera_position": np.array([0, 3, 0]),
            "look_at": np.array([0, 0, 0]),
            "up": np.array([0, 0, -1])
        },
    ]
    return viewpoints


def compute_image_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """두 이미지 간의 정량 지표를 계산합니다.
    
    Args:
        img1: 첫 번째 이미지 (numpy array, uint8, RGB)
        img2: 두 번째 이미지 (numpy array, uint8, RGB)
    
    Returns:
        지표 딕셔너리 (mse, ssim 등)
    """
    metrics = {}
    
    # 이미지를 grayscale로 변환 (SSIM 계산용)
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2).astype(np.float64)
        img2_gray = np.mean(img2, axis=2).astype(np.float64)
    else:
        img1_gray = img1.astype(np.float64)
        img2_gray = img2.astype(np.float64)
    
    # MSE 계산
    mse = np.mean((img1_gray - img2_gray) ** 2)
    metrics["mse"] = float(mse)
    
    # SSIM 계산 (가능한 경우)
    if SSIM_AVAILABLE:
        try:
            ssim_value = ssim(
                img1_gray,
                img2_gray,
                data_range=255.0
            )
            metrics["ssim"] = float(ssim_value)
        except Exception as e:
            print(f"Warning: SSIM 계산 실패: {e}")
            metrics["ssim"] = None
    else:
        metrics["ssim"] = None
    
    return metrics


def evaluate_models(
    baseline_model,
    lora_model,
    device: str,
    image_paths: List[Path],
    category_map: Dict[str, Dict],
    output_dir: str,
    mc_resolution: int = 256
) -> List[Dict]:
    """두 모델을 평가합니다.
    
    Args:
        baseline_model: Baseline TripoSR 모델
        lora_model: LoRA-tuned TripoSR 모델
        device: 디바이스 문자열
        image_paths: 평가할 이미지 경로 리스트
        category_map: 이미지-카테고리 매핑
        output_dir: 출력 디렉토리
        mc_resolution: Marching Cubes 해상도
    
    Returns:
        평가 결과 리스트
    """
    results = []
    viewpoints = get_camera_viewpoints()
    
    for idx, image_path in enumerate(image_paths):
        image_name = image_path.name
        print(f"\n[{idx + 1}/{len(image_paths)}] 평가 중: {image_name}")
        
        if image_name not in category_map:
            print(f"  Warning: {image_name}에 대한 카테고리가 없습니다. 건너뜁니다.")
            continue
        
        try:
            # 이미지 전처리
            processed_image = preprocess_image_for_triposr(str(image_path))
            
            # Baseline 모델로 메쉬 생성
            print("  Baseline 모델로 메쉬 생성 중...")
            baseline_mesh = generate_3d_mesh(
                baseline_model,
                processed_image,
                device,
                mc_resolution=mc_resolution
            )
            
            # LoRA 모델로 메쉬 생성
            print("  LoRA 모델로 메쉬 생성 중...")
            lora_mesh = generate_3d_mesh(
                lora_model,
                processed_image,
                device,
                mc_resolution=mc_resolution
            )
            
            # 메쉬를 원점 중심으로 정규화 (렌더링을 위해)
            baseline_mesh.vertices = baseline_mesh.vertices - baseline_mesh.centroid
            lora_mesh.vertices = lora_mesh.vertices - lora_mesh.centroid
            
            # 각 뷰포인트에서 렌더링 및 비교
            image_metrics = {}
            
            for viewpoint in viewpoints:
                view_name = viewpoint["name"]
                print(f"  {view_name} 뷰 렌더링 중...")
                
                # Baseline 렌더링
                baseline_img = render_mesh_from_viewpoint(
                    baseline_mesh,
                    viewpoint["camera_position"],
                    viewpoint["look_at"],
                    viewpoint["up"]
                )
                
                # LoRA 렌더링
                lora_img = render_mesh_from_viewpoint(
                    lora_mesh,
                    viewpoint["camera_position"],
                    viewpoint["look_at"],
                    viewpoint["up"]
                )
                
                # 이미지 저장
                baseline_path = os.path.join(
                    output_dir,
                    f"{image_path.stem}_baseline_{view_name}.png"
                )
                lora_path = os.path.join(
                    output_dir,
                    f"{image_path.stem}_lora_{view_name}.png"
                )
                
                Image.fromarray(baseline_img).save(baseline_path)
                Image.fromarray(lora_img).save(lora_path)
                
                # 정량 지표 계산 (Baseline과 LoRA 간의 차이)
                metrics = compute_image_metrics(baseline_img, lora_img)
                image_metrics[f"{view_name}_mse"] = metrics["mse"]
                if metrics["ssim"] is not None:
                    image_metrics[f"{view_name}_ssim"] = metrics["ssim"]
            
            # 결과 저장
            result = {
                "image_name": image_name,
                "category": category_map[image_name]["category"],
                "confidence": category_map[image_name]["confidence"],
                **image_metrics
            }
            results.append(result)
            
            print(f"  완료: MSE 평균 = {np.mean([image_metrics.get(f'{v}_mse', 0) for v in ['front', 'side', 'top']]):.4f}")
            
        except Exception as e:
            print(f"  Error: 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def save_results_csv(results: List[Dict], output_path: str):
    """결과를 CSV 파일로 저장합니다.
    
    Args:
        results: 평가 결과 리스트
        output_path: 출력 CSV 파일 경로
    """
    if not results:
        print("Warning: 저장할 결과가 없습니다.")
        return
    
    fieldnames = list(results[0].keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n결과 저장 완료: {output_path}")


def print_results_table(results: List[Dict]):
    """결과를 표 형태로 출력합니다.
    
    Args:
        results: 평가 결과 리스트
    """
    if not results:
        print("출력할 결과가 없습니다.")
        return
    
    print("\n" + "=" * 100)
    print("평가 결과 요약")
    print("=" * 100)
    
    # 헤더
    headers = ["이미지", "카테고리", "Front MSE", "Side MSE", "Top MSE"]
    if SSIM_AVAILABLE:
        headers.extend(["Front SSIM", "Side SSIM", "Top SSIM"])
    
    print(f"{'이미지명':<30} {'카테고리':<15} {'Front MSE':<12} {'Side MSE':<12} {'Top MSE':<12}", end="")
    if SSIM_AVAILABLE:
        print(f" {'Front SSIM':<12} {'Side SSIM':<12} {'Top SSIM':<12}")
    else:
        print()
    
    print("-" * 100)
    
    # 데이터
    for result in results:
        image_name = result["image_name"][:28] + ".." if len(result["image_name"]) > 30 else result["image_name"]
        category = result["category"][:13] + ".." if len(result["category"]) > 15 else result["category"]
        
        print(f"{image_name:<30} {category:<15} ", end="")
        print(f"{result.get('front_mse', 0):<12.4f} ", end="")
        print(f"{result.get('side_mse', 0):<12.4f} ", end="")
        print(f"{result.get('top_mse', 0):<12.4f} ", end="")
        
        if SSIM_AVAILABLE:
            print(f"{result.get('front_ssim', 0):<12.4f} ", end="")
            print(f"{result.get('side_ssim', 0):<12.4f} ", end="")
            print(f"{result.get('top_ssim', 0):<12.4f}")
        else:
            print()
    
    # 평균 계산
    print("-" * 100)
    print("평균:", end="")
    print(f"{'':<45} ", end="")
    
    avg_front_mse = np.mean([r.get('front_mse', 0) for r in results])
    avg_side_mse = np.mean([r.get('side_mse', 0) for r in results])
    avg_top_mse = np.mean([r.get('top_mse', 0) for r in results])
    
    print(f"{avg_front_mse:<12.4f} {avg_side_mse:<12.4f} {avg_top_mse:<12.4f}", end="")
    
    if SSIM_AVAILABLE:
        avg_front_ssim = np.mean([r.get('front_ssim', 0) for r in results])
        avg_side_ssim = np.mean([r.get('side_ssim', 0) for r in results])
        avg_top_ssim = np.mean([r.get('top_ssim', 0) for r in results])
        print(f" {avg_front_ssim:<12.4f} {avg_side_ssim:<12.4f} {avg_top_ssim:<12.4f}")
    else:
        print()
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Baseline vs LoRA TripoSR 모델 비교 평가")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="LoRA 가중치 파일 경로"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="평가할 이미지 개수"
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--mc_resolution",
        type=int,
        default=256,
        help="Marching Cubes 해상도"
    )
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    create_directories()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 카테고리 맵 로드
    print("=" * 60)
    print("카테고리 맵 로드 중...")
    print("=" * 60)
    if not os.path.exists(args.category_map):
        print(f"Error: 카테고리 맵 파일을 찾을 수 없습니다: {args.category_map}")
        return
    
    category_map = load_image_category_map(args.category_map)
    print(f"로드된 이미지-카테고리 매핑: {len(category_map)}개")
    
    # 이미지 경로 로드
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + \
                  list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG"))
    image_paths = sorted(image_paths)
    
    # 카테고리 맵에 있는 이미지만 필터링
    image_paths = [p for p in image_paths if p.name in category_map]
    
    if not image_paths:
        print(f"Error: {args.image_dir}에서 평가할 이미지를 찾을 수 없습니다.")
        return
    
    # 샘플 선택
    if len(image_paths) > args.num_samples:
        import random
        random.seed(42)  # 재현성을 위한 시드 설정
        image_paths = random.sample(image_paths, args.num_samples)
    
    print(f"평가할 이미지 수: {len(image_paths)}개")
    
    # 모델 로드
    print("\n" + "=" * 60)
    print("Baseline 모델 로드 중...")
    print("=" * 60)
    baseline_model, device = load_tripodsr_model()
    
    print("\n" + "=" * 60)
    print("LoRA 모델 로드 중...")
    print("=" * 60)
    lora_base_model, _ = load_tripodsr_model()
    lora_model = load_lora_weights(lora_base_model, args.lora_path, device)
    
    # 평가 수행
    print("\n" + "=" * 60)
    print("평가 시작")
    print("=" * 60)
    results = evaluate_models(
        baseline_model=baseline_model,
        lora_model=lora_model,
        device=device,
        image_paths=image_paths,
        category_map=category_map,
        output_dir=args.output_dir,
        mc_resolution=args.mc_resolution
    )
    
    # 결과 출력 및 저장
    print_results_table(results)
    
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    save_results_csv(results, csv_path)
    
    print("\n" + "=" * 60)
    print("평가 완료!")
    print(f"렌더링 이미지 저장 위치: {args.output_dir}")
    print(f"메트릭 CSV 저장 위치: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
