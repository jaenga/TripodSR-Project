# TripoSR 모델 불러오기
# TripoSR 레포지토리 코드 사용

import sys
import os
from pathlib import Path
from typing import Optional
import torch

# TripoSR 경로 찾기
possible_paths = [
    Path(__file__).parent / "TripoSR",
    Path.cwd() / "TripoSR",
    Path("/content/TripodSR-Project/TripoSR"),
]

TRIPOSR_REPO_PATH = None
for path in possible_paths:
    if path.exists() and (path / "tsr").exists():
        TRIPOSR_REPO_PATH = path
        break

# 없으면 자동으로 클론
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


# TripoSR 모델 불러오기
def load_tripodsr_model(
    device: Optional[str] = None,
    chunk_size: int = 8192,
    pretrained_model_name_or_path: str = "stabilityai/TripoSR",
    config_name: str = "config.yaml",
    weight_name: str = "model.ckpt",
):
    # GPU 자동 감지
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 불러오기
    print(f"TripoSR 모델 로드 중: {pretrained_model_name_or_path}")
    model = TSR.from_pretrained(
        pretrained_model_name_or_path,
        config_name=config_name,
        weight_name=weight_name,
    )

    # 렌더러 설정
    model.renderer.set_chunk_size(chunk_size)

    # GPU로 이동
    model.to(device)
    model.eval()

    print(f"모델 로드 완료. 디바이스: {device}, 청크 크기: {chunk_size}")

    return model, device
