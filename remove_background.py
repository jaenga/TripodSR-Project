# 배경 제거: 이미지 배경 없애고 PNG로 저장

import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import argparse

try:
    from rembg import remove, new_session  # type: ignore
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg가 설치되지 않았습니다. pip install rembg로 설치하세요.")


# 배경 제거하기
def remove_background(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, model_name: str = "u2net"):
    if not REMBG_AVAILABLE:
        raise ImportError("rembg가 설치되지 않았습니다. pip install rembg로 설치하세요.")
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    
    # 출력 경로가 지정되지 않으면 자동 생성
    if output_path is None:
        output_dir = input_path.parent / "no_background"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_no_bg.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"배경 제거 중: {input_path.name} -> {output_path.name}")
    
    # 이미지 읽기
    with open(input_path, 'rb') as f:
        input_data = f.read()
    
    # 세션 생성 및 배경 제거
    session = new_session(model_name)
    output_data = remove(input_data, session=session)
    
    # 결과 저장
    with open(output_path, 'wb') as f:
        f.write(output_data)
    
    print(f"✓ 배경 제거 완료: {output_path}")
    return str(output_path)


# 디렉토리 전체 처리
def process_directory(input_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, model_name: str = "u2net"):
    if not REMBG_AVAILABLE:
        raise ImportError("rembg가 설치되지 않았습니다. pip install rembg로 설치하세요.")
    
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = input_dir / "no_background"
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
    print(f"출력 디렉토리: {output_dir}\n")
    
    # 각 이미지 처리
    success_count = 0
    for idx, image_path in enumerate(sorted(image_paths), 1):
        try:
            output_path = output_dir / f"{image_path.stem}_no_bg.png"
            remove_background(image_path, output_path, model_name=model_name)
            success_count += 1
        except Exception as e:
            print(f"✗ 오류 발생 ({image_path.name}): {e}")
    
    print(f"\n완료: {success_count}/{len(image_paths)}개 이미지 처리 완료")


def main():
    parser = argparse.ArgumentParser(description="이미지 배경 제거 도구")
    parser.add_argument(
        "input",
        type=str,
        help="입력 이미지 파일 또는 디렉토리 경로"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 파일 또는 디렉토리 경로 (지정하지 않으면 자동 생성)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="u2net",
        choices=["u2net", "u2net_human_seg", "u2netp", "silueta", "isnet-general-use"],
        help="rembg 모델 이름 (기본값: u2net)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: 입력 경로를 찾을 수 없습니다: {input_path}")
        return
    
    try:
        if input_path.is_file():
            # 단일 파일 처리
            remove_background(input_path, args.output, args.model)
        elif input_path.is_dir():
            # 디렉토리 처리
            process_directory(input_path, args.output, args.model)
        else:
            print(f"Error: 잘못된 입력 경로입니다: {input_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

