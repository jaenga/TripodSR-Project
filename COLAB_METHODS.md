# 코랩 실행 방법 비교 및 추천

## 📊 방법 비교표

| 방법 | 속도 | 편의성 | 업데이트 | 추천도 |
|------|------|--------|---------|--------|
| **1. GitHub 클론** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **2. ZIP 업로드** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **3. Google Drive 직접** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **4. GitHub + Drive 연동** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🏆 **추천 방법: GitHub 클론 (가장 효율적)**

### 장점
- ✅ **가장 빠름**: `git clone` 한 번이면 끝
- ✅ **버전 관리**: 코드 변경사항 추적 가능
- ✅ **업데이트 용이**: `git pull`로 간단히 업데이트
- ✅ **협업 친화적**: 여러 사람이 동시에 작업 가능
- ✅ **자동화 가능**: 스크립트로 완전 자동화

### 단점
- ⚠️ GitHub 계정 필요
- ⚠️ 인터넷 연결 필요

### 실행 방법

```python
# Colab 노트북 첫 셀
!git clone https://github.com/YOUR_USERNAME/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 환경 설정
from colab_setup import setup_colab_environment, install_requirements
setup_colab_environment()
install_requirements()
```

---

## 🥈 **2순위: GitHub + Google Drive 연동 (데이터 보존)**

### 장점
- ✅ **코드는 GitHub**: 버전 관리 + 빠른 클론
- ✅ **데이터는 Drive**: 대용량 데이터 보존
- ✅ **결과 자동 저장**: 생성된 파일이 Drive에 저장
- ✅ **세션 종료 후에도 유지**: Drive 데이터는 영구 보존

### 실행 방법

```python
# 1. GitHub에서 코드 클론
!git clone https://github.com/YOUR_USERNAME/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 2. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 3. Drive에서 데이터 가져오기 (있는 경우)
import os
if os.path.exists("/content/drive/MyDrive/TripodSR-Project/data"):
    !cp -r /content/drive/MyDrive/TripodSR-Project/data/* data/

# 4. 환경 설정
from colab_setup import setup_colab_environment, install_requirements
setup_colab_environment()
install_requirements()
```

---

## 🥉 **3순위: Google Drive 직접 업로드**

### 장점
- ✅ **설정 간단**: 파일 업로드만 하면 됨
- ✅ **인터넷 없어도 가능**: 이미 업로드된 경우
- ✅ **데이터 보존**: Drive에 저장되어 영구 보존

### 단점
- ⚠️ **업로드 시간**: 대용량 파일은 시간 소요
- ⚠️ **업데이트 불편**: 수동으로 다시 업로드 필요
- ⚠️ **버전 관리 어려움**: Git 히스토리 없음

### 실행 방법

```python
# 1. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. Drive에서 프로젝트 복사
!cp -r /content/drive/MyDrive/TripodSR-Project /content/TripodSR-Project
%cd /content/TripodSR-Project

# 3. 환경 설정
from colab_setup import setup_colab_environment, install_requirements
setup_colab_environment()
install_requirements()
```

---

## ❌ **비추천: ZIP 업로드**

### 단점
- ❌ **매우 느림**: 파일 하나씩 업로드
- ❌ **업데이트 불편**: 전체 다시 업로드 필요
- ❌ **에러 발생 가능**: 파일 누락 위험
- ❌ **비효율적**: 시간 낭비

---

## 🚀 **최종 추천: 하이브리드 방식**

**코드는 GitHub, 데이터는 Google Drive**

```python
# ============================================
# 최적화된 Colab 실행 스크립트
# ============================================

# 1. GitHub에서 코드 클론 (빠름)
!git clone https://github.com/YOUR_USERNAME/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 2. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 3. 환경 설정 및 패키지 설치
from colab_setup import setup_colab_environment, install_requirements, check_gpu_memory
setup_colab_environment()
install_requirements()
check_gpu_memory()

# 4. Drive에서 데이터 가져오기 (있는 경우)
import os
drive_data = "/content/drive/MyDrive/TripodSR-Project/data"
if os.path.exists(drive_data):
    !cp -r {drive_data}/* data/
    print("✓ Drive에서 데이터 복사 완료")
else:
    print("⚠ Drive에 데이터가 없습니다. 직접 업로드하세요.")

# 5. 실행
!python vlm_classifier.py
!python inference.py

# 6. 결과를 Drive에 저장
!mkdir -p /content/drive/MyDrive/TripodSR-Project/outputs
!cp -r outputs/* /content/drive/MyDrive/TripodSR-Project/outputs/
print("✓ 결과가 Drive에 저장되었습니다.")
```

---

## 📝 **실행 체크리스트**

### 첫 실행 전 준비사항

1. **GitHub에 프로젝트 업로드** (선택사항이지만 강력 추천)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/TripodSR-Project.git
   git push -u origin main
   ```

2. **Google Drive에 데이터 업로드** (선택사항)
   - `data/raw_images/` 폴더에 이미지 업로드
   - `data/my_product_dataset/` 폴더에 학습용 이미지 업로드

3. **Colab 노트북 생성**
   - 새 노트북 생성
   - 런타임 → 런타임 유형 변경 → **GPU** 선택

### 매번 실행할 때

1. GitHub에서 최신 코드 클론 (또는 Drive에서 복사)
2. 환경 설정 및 패키지 설치
3. 데이터 준비
4. 실행
5. 결과 다운로드 또는 Drive에 저장

---

## 💡 **팁**

### 1. 코드 업데이트가 있을 때
```python
# GitHub 방식: 간단히 pull
%cd /content/TripodSR-Project
!git pull

# Drive 방식: 다시 복사
!cp -r /content/drive/MyDrive/TripodSR-Project/* /content/TripodSR-Project/
```

### 2. 데이터만 업데이트할 때
```python
# Drive에서 최신 데이터만 가져오기
!cp -r /content/drive/MyDrive/TripodSR-Project/data/* data/
```

### 3. 결과 자동 저장
```python
# inference.py 실행 후 자동으로 Drive에 저장
import shutil
shutil.copytree('outputs', '/content/drive/MyDrive/TripodSR-Project/outputs', dirs_exist_ok=True)
```

---

## 🎯 **결론**

**가장 효율적인 방법: GitHub 클론 + Google Drive 데이터 연동**

- 코드: GitHub에서 클론 (빠르고 버전 관리)
- 데이터: Google Drive에서 가져오기 (대용량 보존)
- 결과: Google Drive에 자동 저장 (영구 보존)

이 방법이 **속도, 편의성, 유지보수성** 모든 면에서 최적입니다! 🚀

