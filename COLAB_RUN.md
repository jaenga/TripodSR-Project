# Colab 실행 가이드 (최종 버전)

## 🚀 빠른 실행 (복사해서 붙여넣기)

### Step 1: 프로젝트 클론 및 환경 설정

```python
# GitHub에서 프로젝트 클론
!git clone https://github.com/jaenga/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 환경 설정 (TripoSR 자동 클론 포함)
from colab_setup import setup_colab_environment, install_requirements, check_gpu_memory

# Google Drive 마운트 + TripoSR 자동 클론 + GPU 확인
setup_colab_environment(mount_drive=True)

# 필수 패키지 설치 (처음 한 번만 실행)
install_requirements()

# GPU 메모리 확인
check_gpu_memory()
```

### Step 2: 데이터 준비 (선택사항)

```python
# 옵션 A: Google Drive에서 데이터 가져오기
import os
drive_data = "/content/drive/MyDrive/TripodSR-Project/data"
if os.path.exists(drive_data):
    !cp -r {drive_data}/* data/ 2>/dev/null || true
    print("✓ Drive에서 데이터 복사 완료")
else:
    print("⚠ Drive에 데이터가 없습니다.")
    print("  data/raw_images/ 디렉토리에 이미지를 업로드하세요.")

# 옵션 B: 직접 업로드
# from google.colab import files
# uploaded = files.upload()
# # 업로드된 파일을 data/raw_images/로 이동
```

### Step 3: 실행

```python
# 이미지 분류
!python vlm_classifier.py

# 3D 모델 생성
!python inference.py
```

### Step 4: 결과 다운로드

```python
from google.colab import files
import zipfile
from pathlib import Path

# GLTF 파일들을 zip으로 압축
output_dir = Path('outputs/gltf_models')
if output_dir.exists():
    with zipfile.ZipFile('outputs.zip', 'w') as zipf:
        for file in output_dir.glob('*.gltf'):
            zipf.write(file, file.name)
    
    # 다운로드
    files.download('outputs.zip')
    
    # Google Drive에도 저장
    !mkdir -p /content/drive/MyDrive/TripodSR-Project/outputs
    !cp -r outputs/* /content/drive/MyDrive/TripodSR-Project/outputs/
    print("✓ 결과가 Drive에 저장되었습니다.")
```

---

## 📋 전체 코드 (한 번에 실행)

```python
# ============================================
# Colab 실행 스크립트 (전체)
# ============================================

# 1. 프로젝트 클론
!git clone https://github.com/jaenga/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 2. 환경 설정 및 패키지 설치
from colab_setup import setup_colab_environment, install_requirements, check_gpu_memory

setup_colab_environment(mount_drive=True)
install_requirements()
check_gpu_memory()

# 3. 데이터 준비 (Drive에서 가져오기 또는 직접 업로드)
import os
drive_data = "/content/drive/MyDrive/TripodSR-Project/data"
if os.path.exists(drive_data):
    !cp -r {drive_data}/* data/ 2>/dev/null || true
    print("✓ Drive에서 데이터 복사 완료")

# 4. 실행
print("\n" + "="*60)
print("이미지 분류 시작...")
print("="*60)
!python vlm_classifier.py

print("\n" + "="*60)
print("3D 모델 생성 시작...")
print("="*60)
!python inference.py

# 5. 결과 다운로드
from google.colab import files
import zipfile
from pathlib import Path

output_dir = Path('outputs/gltf_models')
if output_dir.exists():
    with zipfile.ZipFile('outputs.zip', 'w') as zipf:
        for file in output_dir.glob('*.gltf'):
            zipf.write(file, file.name)
    files.download('outputs.zip')
    print("✓ 다운로드 완료")
```

---

## ⚠️ 주의사항

1. **런타임 설정**: 런타임 → 런타임 유형 변경 → **GPU** 선택 필수
2. **첫 실행**: TripoSR 클론에 1-2분 소요될 수 있습니다
3. **패키지 설치**: `install_requirements()`는 처음 한 번만 실행하면 됩니다
4. **CUDA 경고**: 처음 나오는 CUDA 관련 경고는 무시해도 됩니다

---

## 🔧 문제 해결

### TripoSR 클론 실패 시
```python
# 수동으로 클론
!git clone https://github.com/VAST-AI-Research/TripoSR.git /content/TripodSR-Project/TripoSR
```

### GPU 메모리 부족 시
```python
# chunk_size 줄이기
from triposr_backbone import load_tripodsr_model
model, device = load_tripodsr_model(device="cuda", chunk_size=4096)
```

### 패키지 설치 오류 시
```python
# torchmcubes 재설치
!pip uninstall -y torchmcubes
!pip install git+https://github.com/tatsy/torchmcubes.git
```

