# pip install torch torchvision
# pip install transformers pillow

import os
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

# 디렉토리 생성
def create_directories():
    os.makedirs("data/raw_images", exist_ok=True)
    os.makedirs("data", exist_ok=True)

# 이미지 파일들 불러오기
def load_images(image_dir: str) -> List[str]:
    image_paths = []
    image_paths.extend(Path(image_dir).glob("*.jpg"))
    image_paths.extend(Path(image_dir).glob("*.JPG"))
    image_paths.extend(Path(image_dir).glob("*.jpeg"))
    image_paths.extend(Path(image_dir).glob("*.JPEG"))
    image_paths.extend(Path(image_dir).glob("*.png"))
    image_paths.extend(Path(image_dir).glob("*.PNG"))
    return [str(path) for path in sorted(image_paths)]

# 이미지 분류하기
def classify_images_batch(
    model, 
    processor, 
    image_paths: List[str], 
    candidate_labels: List[str],
    batch_size: int = 16
) -> List[Dict]:
    results = []
    
    # 배치로 나눠서 처리
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # 이미지 열기
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                img = Image.open(path)
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: 이미지를 열 수 없음 {path}: {e}")
                continue
        
        if not images:
            continue
        
        # 모델 입력 준비
        inputs = processor(
            text=candidate_labels,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # 추론 실행
        with torch.no_grad():
            outputs = model(**inputs) 
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1)
        
        # 결과 저장
        for idx, image_path in enumerate(valid_paths):
            image_probs = probs[idx]
            max_idx = image_probs.argmax().item()
            predicted_label = candidate_labels[max_idx]
            confidence = image_probs[max_idx].item()
            
            image_name = os.path.basename(image_path)
            results.append({
                "image_name": image_name,
                "category": predicted_label,
                "confidence": confidence
            })
    
    return results

def main():
    create_directories()
    
    # CLIP 모델 불러오기
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # 이미지 불러오기
    image_dir = "data/raw_images"
    image_paths = load_images(image_dir)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # 분류할 카테고리들
    candidate_labels = ["chair", "shoe", "coffee mug", "car", "bottle"]
    
    # 분류 실행
    print("Classifying images...")
    results = classify_images_batch(
        model=model,
        processor=processor,
        image_paths=image_paths,
        candidate_labels=candidate_labels,
        batch_size=16
    )
    
    # 결과 저장
    output_path = "data/image_category_map.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")
    print(f"Total images classified: {len(results)}")

if __name__ == "__main__":
    main()
