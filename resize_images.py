"""
이미지 해상도 통일 스크립트
모든 이미지를 지정된 해상도로 리사이즈합니다.
"""

import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import argparse


def resize_image(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_size: tuple[int, int] = (512, 512),
    keep_aspect_ratio: bool = True,
    resample: Image.Resampling = Image.Resampling.LANCZOS
):
    """이미지를 지정된 크기로 리사이즈합니다.
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로 (None이면 덮어쓰기)
        target_size: 목표 크기 (width, height)
        keep_aspect_ratio: 종횡비 유지 여부
        resample: 리샘플링 방법
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    image = Image.open(input_path)
    
    # RGBA 이미지 처리
    if image.mode == "RGBA":
        # 알파 채널 유지하면서 리사이즈
        if keep_aspect_ratio:
            image.thumbnail(target_size, resample)
            # 중앙 정렬을 위한 새 이미지 생성
            new_image = Image.new("RGBA", target_size, (255, 255, 255, 0))
            paste_x = (target_size[0] - image.size[0]) // 2
            paste_y = (target_size[1] - image.size[1]) // 2
            new_image.paste(image, (paste_x, paste_y), image)
            image = new_image
        else:
            image = image.resize(target_size, resample)
    else:
        # RGB 이미지 처리
        if keep_aspect_ratio:
            image.thumbnail(target_size, resample)
            # 중앙 정렬을 위한 새 이미지 생성
            new_image = Image.new("RGB", target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.size[0]) // 2
            paste_y = (target_size[1] - image.size[1]) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        else:
            image = image.resize(target_size, resample)
    
    image.save(output_path)
    return str(output_path)


def process_directory(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    target_size: tuple[int, int] = (512, 512),
    keep_aspect_ratio: bool = True
):
    """디렉토리 내의 모든 이미지를 리사이즈합니다."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 지원하는 이미지 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # 이미지 파일 찾기
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
    
    if not image_paths:
        print(f"Warning: {input_dir}에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"발견된 이미지: {len(image_paths)}개")
    print(f"목표 크기: {target_size}")
    print(f"종횡비 유지: {keep_aspect_ratio}\n")
    
    success_count = 0
    for idx, image_path in enumerate(sorted(image_paths), 1):
        try:
            output_path = output_dir / image_path.name
            resize_image(image_path, output_path, target_size, keep_aspect_ratio)
            print(f"[{idx}/{len(image_paths)}] ✓ {image_path.name}")
            success_count += 1
        except Exception as e:
            print(f"[{idx}/{len(image_paths)}] ✗ 오류 발생 ({image_path.name}): {e}")
    
    print(f"\n완료: {success_count}/{len(image_paths)}개 이미지 처리 완료")


def main():
    parser = argparse.ArgumentParser(description="이미지 해상도 통일 도구")
    parser.add_argument("input", type=str, help="입력 이미지 파일 또는 디렉토리 경로")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 파일 또는 디렉토리 경로")
    parser.add_argument("-s", "--size", type=int, nargs=2, default=[512, 512], metavar=("WIDTH", "HEIGHT"), help="목표 크기 (기본값: 512 512)")
    parser.add_argument("--no-aspect-ratio", action="store_true", help="종횡비 유지하지 않음 (기본값: 유지)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: 입력 경로를 찾을 수 없습니다: {input_path}")
        return
    
    target_size = tuple(args.size)
    keep_aspect_ratio = not args.no_aspect_ratio
    
    try:
        if input_path.is_file():
            resize_image(input_path, args.output, target_size, keep_aspect_ratio)
        elif input_path.is_dir():
            process_directory(input_path, args.output, target_size, keep_aspect_ratio)
        else:
            print(f"Error: 잘못된 입력 경로입니다: {input_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

