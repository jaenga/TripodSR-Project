# GitHub 업로드 및 Colab 실행 가이드

## 📤 Step 1: GitHub에 프로젝트 업로드

### 1-1. GitHub 저장소 생성
1. [GitHub](https://github.com)에 로그인
2. 새 저장소 생성 (예: `TripodSR-Project`)
3. **Public** 또는 **Private** 선택

### 1-2. 로컬에서 Git 초기화 및 업로드

```bash
# 프로젝트 디렉토리로 이동
cd /Users/chaewon/Documents/VIL25/TripodSR-Project

# Git 초기화 (이미 되어있다면 생략)
git init

# 모든 파일 추가 (이미지 데이터는 제외됨 - .gitignore에 의해)
git add .

# 커밋
git commit -m "Initial commit: TripodSR project with Colab support"

# GitHub 저장소 연결 (YOUR_USERNAME과 YOUR_REPO_NAME을 실제 값으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/TripodSR-Project.git

# 업로드
git branch -M main
git push -u origin main
```

### 1-3. 확인사항
- ✅ `TripoSR/` 디렉토리가 포함되어 있는지 확인
- ✅ `data/` 디렉토리의 이미지 파일들은 제외됨 (의도된 것)
- ✅ `outputs/` 디렉토리는 제외됨 (의도된 것)

---

## 🚀 Step 2: Colab에서 실행

### 2-1. Colab 노트북 생성
1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성
3. **런타임 → 런타임 유형 변경 → GPU** 선택

### 2-2. 실행 코드

**첫 번째 셀: 프로젝트 클론 및 설정**

```python
# GitHub에서 프로젝트 클론
!git clone https://github.com/YOUR_USERNAME/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 환경 설정 및 패키지 설치
from colab_setup import setup_colab_environment, install_requirements

# Google Drive 마운트 + GPU 확인
setup_colab_environment(mount_drive=True)

# 필수 패키지 설치 (처음 한 번만 실행)
install_requirements()
```

**두 번째 셀: 데이터 준비 (선택사항)**

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

**세 번째 셀: 실행**

```python
# 이미지 분류
!python vlm_classifier.py

# 3D 모델 생성
!python inference.py
```

**네 번째 셀: 결과 다운로드**

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

## ✅ 체크리스트

### GitHub 업로드 전
- [ ] `.gitignore` 파일 확인 (데이터 파일 제외 확인)
- [ ] `TripoSR/` 디렉토리 포함 확인
- [ ] 모든 Python 파일 포함 확인
- [ ] `requirements.txt` 포함 확인

### Colab 실행 전
- [ ] GPU 런타임 선택 확인
- [ ] GitHub 저장소 URL 정확한지 확인
- [ ] 데이터 준비 (Drive 또는 직접 업로드)

### 실행 후
- [ ] GPU 메모리 확인 (약 6GB 필요)
- [ ] 패키지 설치 완료 확인
- [ ] 데이터 파일 존재 확인
- [ ] 결과 파일 생성 확인

---

## 🔧 문제 해결

### 문제 1: `git clone` 실패
```python
# 저장소가 Private인 경우 인증 필요
# 또는 저장소 URL 확인
```

### 문제 2: `TripoSR` 모듈을 찾을 수 없음
```python
# TripoSR 디렉토리가 클론되었는지 확인
!ls -la /content/TripodSR-Project/TripoSR
```

### 문제 3: 패키지 설치 실패
```python
# torchmcubes 재설치
!pip uninstall -y torchmcubes
!pip install git+https://github.com/tatsy/torchmcubes.git
```

### 문제 4: GPU 메모리 부족
```python
# chunk_size 줄이기
from triposr_backbone import load_tripodsr_model
model, device = load_tripodsr_model(device="cuda", chunk_size=4096)
```

---

## 📝 간단 요약

**GitHub 업로드:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/TripodSR-Project.git
git push -u origin main
```

**Colab 실행:**
```python
!git clone https://github.com/YOUR_USERNAME/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project
from colab_setup import setup_colab_environment, install_requirements
setup_colab_environment(mount_drive=True)
install_requirements()
!python vlm_classifier.py
!python inference.py
```

이제 실행하시면 됩니다! 🚀

