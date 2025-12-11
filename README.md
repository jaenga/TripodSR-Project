# TripodSR-Project

AI 기반 2D-to-3D 생성 파이프라인으로, VLM(Vision Language Model) 분류와 LoRA 파인튜닝을 통한 제품 이미지의 3D 모델 생성 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 2D 제품 이미지를 입력받아 3D 모델을 자동으로 생성하는 파이프라인을 제공합니다. 본 프로젝트는 아래 foundation 모델을 기반으로 LoRA fine-tuning 및 파이프라인 확장을 수행하였습니다.

1. **이미지 분류**: CLIP 모델을 사용한 zero-shot 이미지 카테고리 분류
2. **모델 파인튜닝**: LoRA를 사용한 TripodSR 모델 파인튜닝
3. **3D 생성**: 분류된 이미지와 카테고리 정보를 기반으로 3D 모델 생성
4. **GLTF 변환**: 생성된 3D 모델을 GLTF 형식으로 저장

## 🔗 Base Models

- **3D Reconstruction Backbone**: [TripoSR (stabilityai/TripoSR)](https://huggingface.co/stabilityai/TripoSR)
- **Vision-Language Model**: [CLIP (openai/clip-vit-base-patch32)](https://huggingface.co/openai/clip-vit-base-patch32)

## 🗂️ 프로젝트 구조

```
TripodSR-Project/
├── vlm_classifier.py          # CLIP을 사용한 이미지 분류 스크립트
├── train_lora.py              # LoRA 파인튜닝 학습 스크립트
├── inference.py               # 3D 모델 생성 및 GLTF 변환 스크립트
├── requirements.txt           # 필요한 Python 패키지 목록
├── README.md                  # 프로젝트 설명서
│
├── data/                      # 데이터 디렉토리
│   ├── raw_images/           # 원본 이미지 파일들 (.jpg)
│   ├── my_product_dataset/   # 학습용 제품 이미지 데이터셋
│   └── image_category_map.json  # 이미지-카테고리 매핑 JSON 파일
│
├── outputs/                   # 출력 디렉토리
│   └── gltf_models/          # 생성된 GLTF 3D 모델 파일들
│
└── viewer/                    # 3D 모델 뷰어
    ├── index.html
    └── script.js
```

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd TripodSR-Project
```

### 2. 가상 환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 필수 패키지

- `torch>=2.0.0`, `torchvision>=0.15.0` - PyTorch 및 컴퓨터 비전 라이브러리
- `transformers>=4.30.0` - HuggingFace Transformers (CLIP 모델)
- `peft>=0.5.0`, `accelerate>=0.20.0` - LoRA 파인튜닝
- `Pillow>=9.0.0` - 이미지 처리
- `safetensors>=0.3.0` - 모델 가중치 저장
- `open3d>=0.17.0`, `trimesh>=3.15.0` - 3D 처리 및 GLTF 변환

## 📖 사용 방법

### Step 1: 이미지 분류

원본 이미지를 분류하여 카테고리를 추출합니다.

```bash
python vlm_classifier.py
```

**기능:**
- `data/raw_images/`에서 모든 `.jpg` 이미지 로드
- CLIP 모델(`openai/clip-vit-base-patch32`)을 사용한 zero-shot 분류
- 분류 후보: `["chair", "shoe", "coffee mug", "car", "bottle"]`
- 결과를 `data/image_category_map.json`에 저장

**출력 형식:**
```json
[
  {
    "image_name": "example.jpg",
    "category": "chair",
    "confidence": 0.95
  }
]
```

### Step 2: LoRA 파인튜닝 (선택사항)

제품 데이터셋으로 TripodSR 모델을 파인튜닝합니다.

```bash
python train_lora.py
```

**기능:**
- `data/my_product_dataset/`의 이미지로 학습
- 이름에 "attn"이 포함된 레이어에 LoRA 적용 (rank=4, alpha=32)
- Accelerate를 사용한 단일 GPU 학습 (fp16 mixed precision)
- Adam 옵티마이저 (learning rate=1e-4)
- LoRA 어댑터 가중치를 `/content/drive/MyDrive/tripodsr/checkpoints/lora_weights.safetensors`에 저장

**주의:** `train_lora.py`의 `load_tripodsr_model()` 함수를 실제 TripodSR 모델 로딩 로직으로 교체해야 합니다.

### Step 3: 3D 모델 생성

분류된 이미지로부터 3D 모델을 생성하고 GLTF 형식으로 저장합니다.

```bash
python inference.py
```

**기능:**
- TripodSR 베이스 모델 로드
- LoRA 가중치 로드 및 병합 (존재하는 경우)
- `data/image_category_map.json`에서 이미지-카테고리 매핑 로드
- 각 이미지에 대해 카테고리를 조건부 프롬프트로 사용하여 3D 생성
- 생성된 3D 모델을 GLTF 형식으로 변환
- 결과를 `outputs/gltf_models/<image_name>.gltf`에 저장

**주의:** `inference.py`의 `load_tripodsr_model()` 및 `model.generate_3d()` 함수를 실제 구현으로 교체해야 합니다.

## ⚙️ 설정 및 커스터마이징

### 이미지 분류 카테고리 변경

`vlm_classifier.py`에서 후보 레이블을 수정할 수 있습니다:

```python
candidate_labels = ["chair", "shoe", "coffee mug", "car", "bottle"]
```

### LoRA 하이퍼파라미터 조정

`train_lora.py`에서 LoRA 설정을 변경할 수 있습니다:

```python
model = apply_lora_to_model(model, r=4, alpha=32)
```

- `r`: LoRA rank (기본값: 4)
- `alpha`: LoRA alpha (기본값: 32)

### 학습 파라미터 조정

`train_lora.py`에서 학습 설정을 변경할 수 있습니다:

- 배치 크기: `DataLoader`의 `batch_size` 파라미터
- 학습률: `torch.optim.Adam`의 `lr` 파라미터
- 에폭 수: `num_epochs` 변수

## 🔧 주요 함수 설명

### vlm_classifier.py

- `load_images()`: 디렉토리에서 이미지 경로 로드
- `classify_images_batch()`: 배치 단위로 이미지 분류 수행
- `main()`: 전체 분류 파이프라인 실행

### train_lora.py

- `apply_lora_to_model()`: 모델에 LoRA 적용
- `ImageDataset`: 이미지 데이터셋 클래스
- `train_epoch()`: 한 에폭 학습
- `save_lora_weights()`: LoRA 가중치만 저장

### inference.py

- `load_tripodsr_model()`: TripodSR 베이스 모델 로드
- `load_lora_weights()`: LoRA 가중치 로드 및 병합
- `load_image_category_map()`: 이미지-카테고리 매핑 로드
- `mesh_to_gltf()`: 메쉬를 GLTF 형식으로 변환

## 📝 참고사항

1. **모델 구현 필요**: 현재 스크립트에는 예제 TripodSR 모델 구현이 포함되어 있습니다. 실제 TripodSR 모델로 교체해야 합니다.

2. **GPU/CPU**: 
   - GPU가 없어도 CPU로 실행 가능하도록 구현되어 있습니다.
   - CUDA 사용 시 자동으로 감지하여 GPU를 사용합니다.

3. **디렉토리 생성**: 필요한 디렉토리는 스크립트 실행 시 자동으로 생성됩니다.

4. **LoRA 가중치 경로**: 학습 시 저장된 LoRA 가중치 경로가 추론 스크립트와 일치해야 합니다.

