# Google Drive 폴더 구조 가이드

## 📁 필요한 폴더 구조

Google Drive에 다음과 같은 구조로 데이터를 준비하세요:

```
Google Drive/
└── MyDrive/
    ├── TripodSR-Project/          # (선택사항) 프로젝트 백업용
    │   └── data/
    │       ├── raw_images/        # 원본 이미지들 (추론용)
    │       │   ├── image1.jpg
    │       │   ├── image2.jpg
    │       │   └── ...
    │       ├── my_product_dataset/ # 학습용 이미지들 (LoRA 학습용)
    │       │   ├── train1.jpg
    │       │   ├── train2.jpg
    │       │   └── ...
    │       └── image_category_map.json  # (자동 생성됨)
    │
    └── tripodsr/                   # LoRA 가중치 저장용
        └── checkpoints/
            └── lora_weights.safetensors  # (학습 후 자동 생성됨)
```

## 📋 각 폴더 설명

### 1. `/content/drive/MyDrive/TripodSR-Project/data/raw_images/`
- **용도**: 추론(inference)에 사용할 원본 이미지
- **형식**: `.jpg`, `.JPG`, `.png`, `.PNG` 등
- **필수 여부**: ✅ 필수 (inference.py 실행 시 필요)

### 2. `/content/drive/MyDrive/TripodSR-Project/data/my_product_dataset/`
- **용도**: LoRA 학습에 사용할 학습용 이미지
- **형식**: `.jpg`, `.JPG`, `.png`, `.PNG` 등
- **필수 여부**: ⚠️ LoRA 학습 시에만 필요
- **참고**: `raw_images`와 동일한 이미지를 사용해도 됨

### 3. `/content/drive/MyDrive/TripodSR-Project/data/image_category_map.json`
- **용도**: 이미지-카테고리 매핑 정보
- **생성**: `vlm_classifier.py` 실행 시 자동 생성
- **형식**:
```json
[
  {
    "image_name": "image1.jpg",
    "category": "chair",
    "confidence": 0.95
  }
]
```

### 4. `/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors`
- **용도**: 학습된 LoRA 가중치 저장
- **생성**: `train_lora.py` 실행 시 자동 생성
- **필수 여부**: ⚠️ LoRA 사용 시에만 필요

## 🚀 Colab에서 사용하기

### 데이터 가져오기

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 데이터 복사
import os
import shutil

# raw_images 복사
drive_raw = "/content/drive/MyDrive/TripodSR-Project/data/raw_images"
if os.path.exists(drive_raw):
    !mkdir -p data/raw_images
    !cp -r {drive_raw}/* data/raw_images/
    print("✓ raw_images 복사 완료")

# 학습용 데이터 복사
drive_train = "/content/drive/MyDrive/TripodSR-Project/data/my_product_dataset"
if os.path.exists(drive_train):
    !mkdir -p data/my_product_dataset
    !cp -r {drive_train}/* data/my_product_dataset/
    print("✓ 학습용 데이터 복사 완료")
```

## 📝 최소 필수 구조

LoRA 학습 없이 추론만 하려면:

```
Google Drive/MyDrive/TripodSR-Project/data/raw_images/
├── image1.jpg
├── image2.jpg
└── ...
```

LoRA 학습까지 하려면:

```
Google Drive/MyDrive/TripodSR-Project/data/
├── raw_images/          # 추론용
│   └── *.jpg
└── my_product_dataset/  # 학습용
    └── *.jpg
```

## 💡 팁

1. **같은 이미지 사용 가능**: `raw_images`와 `my_product_dataset`에 같은 이미지를 넣어도 됩니다.

2. **자동 생성 파일**: `image_category_map.json`은 `vlm_classifier.py` 실행 시 자동 생성됩니다.

3. **LoRA 가중치**: 학습 후 자동으로 Drive에 저장되므로 수동 업로드 불필요합니다.

4. **폴더 자동 생성**: Colab에서 실행 시 필요한 폴더는 자동으로 생성됩니다.

