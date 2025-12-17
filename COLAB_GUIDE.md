# Google Colab 실행 가이드

이 가이드는 Google Colab에서 TripodSR 프로젝트를 실행하는 방법을 설명합니다.

## 🚀 빠른 시작

### 1단계: Colab 노트북 생성

1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성
3. **런타임 → 런타임 유형 변경 → GPU** 선택 (T4 또는 V100 권장)

### 2단계: 프로젝트 설정

첫 번째 셀에 다음 코드를 실행:

```python
# 프로젝트 클론 또는 업로드
!git clone https://github.com/your-repo/TripodSR-Project.git
# 또는 Google Drive에 업로드한 경우:
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/TripodSR-Project

%cd /content/TripodSR-Project
```

### 3단계: 환경 설정 및 패키지 설치

```python
# Colab 환경 설정
from colab_setup import setup_colab_environment, install_requirements, check_gpu_memory

# 환경 설정 (Google Drive 마운트 포함)
setup_colab_environment()

# 패키지 설치 (처음 한 번만)
install_requirements()

# GPU 메모리 확인
check_gpu_memory()
```

### 4단계: 데이터 준비

**옵션 A: Google Drive에 데이터 업로드**
```python
# Google Drive에 다음 구조로 데이터 업로드:
# /content/drive/MyDrive/TripodSR-Project/
#   ├── data/
#   │   ├── raw_images/        # 원본 이미지들
#   │   └── my_product_dataset/ # 학습용 이미지들 (선택)
#   └── (프로젝트 파일들)

# 심볼릭 링크 생성
import os
os.makedirs("data/raw_images", exist_ok=True)
os.makedirs("data/my_product_dataset", exist_ok=True)
```

**옵션 B: 직접 업로드**
```python
from google.colab import files
# 이미지 파일들을 업로드하고 data/raw_images/로 이동
```

### 5단계: 실행

**Step 1: 이미지 분류**
```python
!python vlm_classifier.py
```

**Step 2: LoRA 학습 (선택사항)**
```python
!python train_lora.py
```

**Step 3: 3D 모델 생성**
```python
!python inference.py
```

## 📋 전체 실행 예제

```python
# ============================================
# 1. 환경 설정
# ============================================
from colab_setup import setup_colab_environment, install_requirements, check_gpu_memory

setup_colab_environment()
install_requirements()  # 처음 한 번만
check_gpu_memory()

# ============================================
# 2. 이미지 분류
# ============================================
!python vlm_classifier.py

# ============================================
# 3. LoRA 학습 (선택사항)
# ============================================
# !python train_lora.py

# ============================================
# 4. 3D 모델 생성
# ============================================
!python inference.py

# ============================================
# 5. 결과 확인 및 다운로드
# ============================================
from google.colab import files
import zipfile

# GLTF 파일들을 zip으로 압축
with zipfile.ZipFile('outputs.zip', 'w') as zipf:
    for file in Path('outputs/gltf_models').glob('*.gltf'):
        zipf.write(file)

# 다운로드
files.download('outputs.zip')
```

## ⚙️ GPU 메모리 최적화

Colab의 무료 GPU(T4)는 약 15GB VRAM을 제공하지만, TripoSR은 약 6GB가 필요합니다.

메모리가 부족한 경우:

```python
# triposr_backbone.py의 chunk_size를 줄이기
from triposr_backbone import load_tripodsr_model

model, device = load_tripodsr_model(
    device="cuda",
    chunk_size=4096  # 기본값 8192에서 줄임
)
```

## 🔧 문제 해결

### 문제 1: GPU가 할당되지 않음
- **해결**: 런타임 → 런타임 유형 변경 → GPU 선택
- 또는 다른 세션 종료 후 재시도

### 문제 2: Google Drive 마운트 실패
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 문제 3: 패키지 설치 오류
```python
# torchmcubes 재설치
!pip uninstall -y torchmcubes
!pip install git+https://github.com/tatsy/torchmcubes.git
```

### 문제 4: CUDA 버전 불일치
```python
# PyTorch CUDA 버전 확인
import torch
print(torch.version.cuda)

# 필요시 재설치
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 예상 실행 시간

- **이미지 분류**: 이미지당 ~0.1초 (CLIP 모델)
- **LoRA 학습**: 에폭당 ~5-10분 (데이터셋 크기에 따라)
- **3D 생성**: 이미지당 ~1-3초 (GPU에 따라)

## 💡 팁

1. **세션 유지**: Colab 세션은 약 12시간 후 자동 종료됩니다. 중요한 작업은 Google Drive에 저장하세요.

2. **배치 처리**: 여러 이미지를 한 번에 처리할 수 있습니다.

3. **결과 저장**: 생성된 GLTF 파일은 Google Drive에 자동으로 저장됩니다.

4. **Pro 사용**: Colab Pro($10/월)를 사용하면 더 나은 GPU와 더 긴 세션 시간을 얻을 수 있습니다.

## 📝 주의사항

- ⚠️ Colab 무료 버전은 GPU 사용 시간이 제한됩니다 (일일 할당량)
- ⚠️ 세션이 종료되면 `/content/` 디렉토리의 데이터는 삭제됩니다
- ✅ 중요한 파일은 항상 Google Drive에 저장하세요
- ✅ LoRA 가중치는 자동으로 Google Drive에 저장됩니다

