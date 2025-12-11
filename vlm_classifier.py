# pip install torch torchvision
# pip install transformers pillow

import os
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

# 디렉토리 생성 함수 정의
def create_directories():
    # 필수 데이터 디렉토리가 없으면 자동 생성
    os.makedirs("data/raw_images", exist_ok=True) # 원본 이미지 디렉토리
    os.makedirs("data", exist_ok=True) # 기본 데이터 디렉토리

# 이미지 로드 함수 정의
def load_images(image_dir: str) -> List[str]:
    # .jpg 및 .JPG 확장자를 가진 이미지 파일 경로 목록 생성
    image_paths = list(Path(image_dir).glob("*.jpg"))
    image_paths.extend(Path(image_dir).glob("*.JPG"))
    return [str(path) for path in sorted(image_paths)]

# 이미지 분류 함수 정의
def classify_images_batch(
    model, 
    processor, 
    image_paths: List[str], 
    candidate_labels: List[str], # 분류 후보 라벨 목록
    batch_size: int = 16 # 배치 크기
) -> List[Dict]:
    # 제로-샷 분류 결과 목록
    results = []
    
    # 배치 단위로 이미지 처리 (0, 16, 32 ...)
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # 이미지 로드
        images = [] # 실제 이미지 PIL 객체 목록
        valid_paths = [] # 성공적으로 로드된 이미지 경로 문자열
        for path in batch_paths:
            try:
                img = Image.open(path) # PIL 객체로 이미지 로드
                images.append(img) # 이미지 객체 목록에 추가
                valid_paths.append(path) # 성공적으로 로드된 이미지 경로 목록에 추가
            except Exception as e:
                print(f"Warning: 이미지를 열 수 없음 {path}: {e}")
                continue
        
        if not images:
            continue
        
        # 이미지와 텍스트 입력 준비
        inputs = processor(
            text=candidate_labels,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # CPU에서 추론 실행 (gradient 계산 비활성화)
        with torch.no_grad():
            outputs = model(**inputs) 
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1) # 소프트맥스 함수를 사용하여 확률 분포 계산
        
        # 각 이미지에 대해 예측 추출
        for idx, image_path in enumerate(valid_paths):
            image_probs = probs[idx] # 각 이미지에 대한 확률 분포
            max_idx = image_probs.argmax().item() # 가장 높은 확률을 가진 라벨 인덱스
            predicted_label = candidate_labels[max_idx] # 예측된 라벨
            confidence = image_probs[max_idx].item() # 예측된 라벨의 확률
            
            image_name = os.path.basename(image_path) # 이미지 파일 이름
            results.append({ # 결과 목록에 추가
                "image_name": image_name,
                "category": predicted_label,
                "confidence": confidence
            })
    
    return results

def main():
    # 디렉토리 생성
    create_directories()
    
    # CLIP 모델 로드
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name) # CLIP 모델 로드
    processor = CLIPProcessor.from_pretrained(model_name) # CLIP 프로세서 로드
    model.eval() # 모델을 평가 모드로 설정
    
    # 이미지 디렉토리에서 모든 이미지 로드
    image_dir = "data/raw_images"
    image_paths = load_images(image_dir)
    
    if not image_paths:
        print(f"No .jpg images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # 분류 후보 라벨 목록
    candidate_labels = ["chair", "shoe", "coffee mug", "car", "bottle"]
    
    # 이미지 분류
    print("Classifying images...")
    results = classify_images_batch(
        model=model,
        processor=processor,
        image_paths=image_paths,
        candidate_labels=candidate_labels,
        batch_size=16
    )
    
    # 결과를 JSON 파일로 저장
    output_path = "data/image_category_map.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")
    print(f"Total images classified: {len(results)}")

if __name__ == "__main__":
    main()
