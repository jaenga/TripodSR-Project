"""
TripoSR 모델 로딩을 위한 백본 모듈

이 모듈은 실제 TripoSR 모델을 로드하고 초기화하는 함수를 제공합니다.
로컬에 클론된 TripoSR 레포지토리의 코드를 사용합니다.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import torch

# TripoSR 레포지토리 경로를 sys.path에 추가
# 여러 가능한 경로를 시도
possible_paths = [
    Path(__file__).parent / "TripoSR",  # 현재 파일 기준
    Path.cwd() / "TripoSR",  # 현재 작업 디렉토리 기준
    Path("/content/TripodSR-Project/TripoSR"),  # Colab 기본 경로
]

TRIPOSR_REPO_PATH = None
for path in possible_paths:
    if path.exists() and (path / "tsr").exists():
        TRIPOSR_REPO_PATH = path
        break

# TripoSR 디렉토리를 찾지 못한 경우 자동으로 클론 시도
if TRIPOSR_REPO_PATH is None:
    print("⚠ TripoSR 디렉토리를 찾을 수 없습니다. 자동으로 클론을 시도합니다...")
    
    # 프로젝트 루트 찾기
    project_root = Path(__file__).parent
    triposr_path = project_root / "TripoSR"
    
    # Colab 환경인지 확인
    try:
        import google.colab  # type: ignore
        is_colab_env = True
    except ImportError:
        is_colab_env = False
    
    if is_colab_env:
        # Colab에서 TripoSR 클론
        import subprocess
        print(f"GitHub에서 TripoSR 클론 중: {triposr_path}")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git", str(triposr_path)],
                check=True,
                capture_output=True,
                text=True
            )
            if triposr_path.exists() and (triposr_path / "tsr").exists():
                TRIPOSR_REPO_PATH = triposr_path
                print("✓ TripoSR 클론 완료")
            else:
                raise FileNotFoundError("TripoSR 클론 후에도 tsr 디렉토리를 찾을 수 없습니다.")
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(
                f"TripoSR 클론 실패: {e}\n"
                f"수동으로 클론하려면: git clone https://github.com/VAST-AI-Research/TripoSR.git {triposr_path}"
            )
    else:
        # 로컬 환경에서는 오류 발생
        raise FileNotFoundError(
            f"TripoSR 디렉토리를 찾을 수 없습니다.\n"
            f"다음 명령어로 클론하세요:\n"
            f"  git clone https://github.com/VAST-AI-Research/TripoSR.git {triposr_path}"
        )

# sys.path에 추가 (중복 방지)
triposr_path_str = str(TRIPOSR_REPO_PATH)
if triposr_path_str not in sys.path:
    sys.path.insert(0, triposr_path_str)

# tsr 모듈 import
try:
    from tsr.system import TSR  # type: ignore
except ImportError as e:
    raise ImportError(
        f"tsr 모듈을 import할 수 없습니다. "
        f"TripoSR 경로: {TRIPOSR_REPO_PATH}\n"
        f"원본 오류: {e}"
    )


def load_tripodsr_model(
    device: Optional[str] = None,
    chunk_size: int = 8192,
    pretrained_model_name_or_path: str = "stabilityai/TripoSR",
    config_name: str = "config.yaml",
    weight_name: str = "model.ckpt",
):
    """
    HuggingFace 'stabilityai/TripoSR' 체크포인트를 로드해서
    TSR 모델 인스턴스를 반환하는 함수.

    Args:
        device: 사용할 디바이스 ('cuda', 'cpu' 등). None이면 자동 감지.
               CUDA가 사용 가능하면 'cuda', 아니면 'cpu'를 사용합니다.
        chunk_size: 렌더링 시 사용할 청크 크기. VRAM 사용량과 속도의 균형을 조절합니다.
                   기본값: 8192 (더 작은 값은 VRAM 사용량 감소, 속도 저하)
        pretrained_model_name_or_path: 사전 학습된 모델 경로 또는 HuggingFace 모델 ID.
                                      기본값: "stabilityai/TripoSR"
        config_name: 설정 파일 이름. 기본값: "config.yaml"
        weight_name: 가중치 파일 이름. 기본값: "model.ckpt"

    Returns:
        tuple: (model, device)
            - model: 로드된 TSR 모델 인스턴스
            - device: 실제 사용된 디바이스 문자열

    Example:
        >>> model, device = load_tripodsr_model()
        >>> model, device = load_tripodsr_model(device="cuda", chunk_size=4096)
    """
    # 디바이스 자동 감지
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    print(f"TripoSR 모델 로드 중: {pretrained_model_name_or_path}")
    model = TSR.from_pretrained(
        pretrained_model_name_or_path,
        config_name=config_name,
        weight_name=weight_name,
    )

    # 렌더러 청크 크기 설정
    model.renderer.set_chunk_size(chunk_size)

    # 모델을 디바이스로 이동
    model.to(device)
    model.eval()  # 추론 모드로 설정

    print(f"모델 로드 완료. 디바이스: {device}, 청크 크기: {chunk_size}")

    return model, device
