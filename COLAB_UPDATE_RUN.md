# 코랩 업데이트 실행 가이드

## 🔄 이미 프로젝트가 클론되어 있는 경우 (업데이트만)

```python
# 1. 프로젝트 디렉토리로 이동
%cd /content/TripodSR-Project

# 2. 최신 코드 가져오기
!git pull origin main

# 3. 배경 제거 이미지가 이미 있으면 그대로 사용
#    (없으면 아래 "배경 제거 다시 실행" 참고)

# 4. inference.py 실행 (개선된 배경 처리 로직 적용됨)
!python inference.py
```

## 🆕 처음부터 실행하는 경우

```python
# 1. 프로젝트 클론
!git clone https://github.com/jaenga/TripodSR-Project.git /content/TripodSR-Project
%cd /content/TripodSR-Project

# 2. 환경 설정
from colab_setup import setup_colab_environment, install_requirements
setup_colab_environment(mount_drive=True)
install_requirements()

# 3. 데이터 준비 (Drive에서 가져오기)
import os
drive_data = "/content/drive/MyDrive/TripodSR-Project/data"
if os.path.exists(drive_data):
    !cp -r {drive_data}/* data/ 2>/dev/null || true
    print("✓ Drive에서 데이터 복사 완료")

# 4. 이미지 분류 (처음 한 번만)
!python vlm_classifier.py

# 5. 3D 모델 생성 (개선된 배경 처리 적용)
!python inference.py
```

## 🎨 배경 제거 다시 실행 (선택사항)

배경 제거 품질이 만족스럽지 않으면 다시 실행할 수 있습니다:

```python
# 배경 제거 스크립트 실행
!python remove_background.py data/raw_images/my_mug_1.jpeg -m u2net
!python remove_background.py data/raw_images/my_mug_2.jpeg -m u2net
!python remove_background.py data/raw_images/my_mug_3.jpeg -m u2net
!python remove_background.py data/raw_images/my_mug_4.jpeg -m u2net
!python remove_background.py data/raw_images/my_mug_5.jpeg -m u2net

# 또는 전체 디렉토리 처리
!python remove_background.py data/raw_images/ -m u2net
```

## ✅ 권장 실행 순서

**이미 배경 제거된 이미지가 있는 경우:**
1. `git pull`로 최신 코드 받기
2. `inference.py` 실행

**배경 제거를 다시 하고 싶은 경우:**
1. `git pull`로 최신 코드 받기
2. `remove_background.py` 실행 (선택사항)
3. `inference.py` 실행

## 📝 주요 개선 사항

- ✅ Alpha threshold 적용으로 더 정확한 배경 제거
- ✅ GLTF 인덱스 범위 초과 오류 자동 수정
- ✅ 메쉬 검증 및 자동 복구

