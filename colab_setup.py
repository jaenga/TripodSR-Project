"""
Colab 환경 설정 및 유틸리티 함수

이 모듈은 Google Colab 환경에서 프로젝트를 실행하기 위한 설정을 제공합니다.
"""

import os
import sys
from pathlib import Path

def is_colab():
    """Colab 환경인지 확인"""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False

def setup_colab_environment(mount_drive=True, workspace_path="/content/TripodSR-Project"):
    """Colab 환경 설정
    
    Args:
        mount_drive: Google Drive 마운트 여부 (기본값: True)
        workspace_path: 작업 디렉토리 경로 (기본값: /content/TripodSR-Project)
    """
    if not is_colab():
        print("로컬 환경입니다. Colab 설정을 건너뜁니다.")
        return False
    
    print("=" * 60)
    print("Google Colab 환경 감지됨")
    print("=" * 60)
    
    # Google Drive 마운트
    if mount_drive:
        try:
            from google.colab import drive  # type: ignore
            print("\nGoogle Drive 마운트 중...")
            drive.mount('/content/drive', force_remount=False)
            print("✓ Google Drive 마운트 완료")
        except Exception as e:
            print(f"⚠ Google Drive 마운트 실패: {e}")
            print("수동으로 마운트하려면: from google.colab import drive; drive.mount('/content/drive')")
    
    # 작업 디렉토리 설정
    if not os.path.exists(workspace_path):
        print(f"\n작업 디렉토리 생성: {workspace_path}")
        os.makedirs(workspace_path, exist_ok=True)
    
    os.chdir(workspace_path)
    print(f"✓ 작업 디렉토리: {os.getcwd()}")
    
    # TripoSR 디렉토리 확인 및 자동 클론
    triposr_path = Path(workspace_path) / "TripoSR"
    if not triposr_path.exists() or not (triposr_path / "tsr").exists():
        print("\n⚠ TripoSR 디렉토리가 없습니다. 자동으로 클론합니다...")
        
        # 기존 디렉토리가 있지만 비어있거나 손상된 경우 삭제
        if triposr_path.exists():
            print(f"  기존 TripoSR 디렉토리 삭제 중...")
            import shutil
            try:
                shutil.rmtree(triposr_path)
            except Exception as e:
                print(f"  ⚠ 디렉토리 삭제 실패: {e}")
        
        # Git 클론
        import subprocess
        print(f"  GitHub에서 TripoSR 클론 중...")
        try:
            result = subprocess.run(
                ["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git", str(triposr_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            if triposr_path.exists() and (triposr_path / "tsr").exists():
                print("✓ TripoSR 클론 완료")
            else:
                raise FileNotFoundError("클론 후에도 tsr 디렉토리를 찾을 수 없습니다.")
        except subprocess.TimeoutExpired:
            print("⚠ TripoSR 클론 시간 초과 (5분)")
            print("수동으로 클론하려면:")
            print(f"  !rm -rf {triposr_path}")
            print(f"  !git clone https://github.com/VAST-AI-Research/TripoSR.git {triposr_path}")
        except subprocess.CalledProcessError as e:
            print(f"⚠ TripoSR 클론 실패 (exit code: {e.returncode})")
            print(f"  오류 메시지: {e.stderr}")
            print("\n수동으로 클론하려면 다음 명령어를 실행하세요:")
            print(f"  !rm -rf {triposr_path}")
            print(f"  !git clone https://github.com/VAST-AI-Research/TripoSR.git {triposr_path}")
        except Exception as e:
            print(f"⚠ TripoSR 클론 중 오류 발생: {e}")
            print("\n수동으로 클론하려면 다음 명령어를 실행하세요:")
            print(f"  !rm -rf {triposr_path}")
            print(f"  !git clone https://github.com/VAST-AI-Research/TripoSR.git {triposr_path}")
    
    # GPU 확인
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    print("=" * 60)
    return True

def install_requirements():
    """필요한 패키지 설치"""
    print("\n필수 패키지 설치 중...")
    
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "accelerate>=0.20.0",
        "Pillow>=9.0.0",
        "safetensors>=0.3.0",
        "open3d>=0.17.0",
        "trimesh>=3.15.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "huggingface-hub>=0.16.0",
    ]
    
    # torchmcubes는 별도 설치
    print("torchmcubes 설치 중...")
    os.system("pip install git+https://github.com/tatsy/torchmcubes.git")
    
    for package in packages:
        print(f"  설치 중: {package}")
        os.system(f"pip install -q {package}")
    
    print("✓ 패키지 설치 완료")

def check_gpu_memory():
    """GPU 메모리 확인"""
    import torch
    if not torch.cuda.is_available():
        return None
    
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    free = total_memory - cached
    
    print(f"\nGPU 메모리 상태:")
    print(f"  총 메모리: {total_memory:.2f} GB")
    print(f"  사용 중: {cached:.2f} GB")
    print(f"  여유: {free:.2f} GB")
    
    if free < 6:
        print(f"\n⚠ 경고: 여유 메모리가 6GB 미만입니다.")
        print(f"  TripoSR은 약 6GB VRAM이 필요합니다.")
        print(f"  chunk_size를 줄이거나 다른 세션을 종료하세요.")
    
    return {
        "total": total_memory,
        "allocated": allocated,
        "cached": cached,
        "free": free
    }

def clone_from_github(repo_url, target_path="/content/TripodSR-Project"):
    """GitHub에서 프로젝트 클론"""
    if not is_colab():
        print("로컬 환경에서는 git clone을 직접 사용하세요.")
        return False
    
    print(f"GitHub에서 프로젝트 클론 중: {repo_url}")
    os.system(f"git clone {repo_url} {target_path}")
    
    if os.path.exists(target_path):
        print(f"✓ 클론 완료: {target_path}")
        os.chdir(target_path)
        return True
    else:
        print("⚠ 클론 실패")
        return False

def sync_from_drive(source_path="/content/drive/MyDrive/TripodSR-Project", 
                    target_path="/content/TripodSR-Project"):
    """Google Drive에서 프로젝트 동기화"""
    if not is_colab():
        print("로컬 환경에서는 직접 복사하세요.")
        return False
    
    if not os.path.exists(source_path):
        print(f"⚠ Drive 경로를 찾을 수 없습니다: {source_path}")
        return False
    
    print(f"Drive에서 프로젝트 동기화 중...")
    os.system(f"cp -r {source_path}/* {target_path}/")
    print(f"✓ 동기화 완료")
    return True

if __name__ == "__main__":
    # Colab 환경 설정
    setup_colab_environment()
    
    # 패키지 설치 (필요시)
    if is_colab():
        install_requirements()
        check_gpu_memory()
