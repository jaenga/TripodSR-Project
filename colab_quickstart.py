"""
Colab 빠른 시작 스크립트

이 스크립트를 Colab에서 실행하면 자동으로 프로젝트를 설정합니다.
"""

from colab_setup import (
    is_colab, 
    setup_colab_environment, 
    install_requirements, 
    check_gpu_memory,
    clone_from_github,
    sync_from_drive
)

def quick_start(method="github", github_url=None, use_drive=True):
    """
    빠른 시작 함수
    
    Args:
        method: "github" 또는 "drive"
        github_url: GitHub 저장소 URL (method="github"일 때 필요)
        use_drive: Google Drive 사용 여부
    """
    if not is_colab():
        print("⚠ 이 스크립트는 Colab 환경에서만 실행할 수 있습니다.")
        return
    
    print("=" * 70)
    print("🚀 TripodSR 프로젝트 빠른 시작")
    print("=" * 70)
    
    # 1. 환경 설정
    print("\n[1/5] 환경 설정 중...")
    setup_colab_environment(mount_drive=use_drive)
    
    # 2. 프로젝트 가져오기
    print("\n[2/5] 프로젝트 가져오기 중...")
    if method == "github":
        if github_url is None:
            github_url = input("GitHub 저장소 URL을 입력하세요: ")
        clone_from_github(github_url)
    elif method == "drive":
        sync_from_drive()
    else:
        print("⚠ 지원하지 않는 방법입니다. 'github' 또는 'drive'를 사용하세요.")
        return
    
    # 3. 패키지 설치
    print("\n[3/5] 패키지 설치 중...")
    install_requirements()
    
    # 4. GPU 확인
    print("\n[4/5] GPU 확인 중...")
    check_gpu_memory()
    
    # 5. 데이터 확인
    print("\n[5/5] 데이터 확인 중...")
    import os
    from pathlib import Path
    
    data_paths = {
        "raw_images": "data/raw_images",
        "product_dataset": "data/my_product_dataset",
        "category_map": "data/image_category_map.json"
    }
    
    print("\n데이터 상태:")
    for name, path in data_paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len(list(Path(path).glob("*")))
                print(f"  ✓ {name}: {file_count}개 파일")
            else:
                print(f"  ✓ {name}: 파일 존재")
        else:
            print(f"  ⚠ {name}: 없음")
    
    # Drive에서 데이터 가져오기 (있는 경우)
    if use_drive:
        drive_data = "/content/drive/MyDrive/TripodSR-Project/data"
        if os.path.exists(drive_data):
            print("\nGoogle Drive에서 데이터 복사 중...")
            os.system(f"cp -r {drive_data}/* data/ 2>/dev/null || true")
            print("✓ 데이터 복사 완료")
    
    print("\n" + "=" * 70)
    print("✅ 설정 완료! 이제 다음 명령어를 실행하세요:")
    print("=" * 70)
    print("\n1. 이미지 분류:")
    print("   !python vlm_classifier.py")
    print("\n2. 3D 모델 생성:")
    print("   !python inference.py")
    print("\n3. 결과 다운로드:")
    print("   from google.colab import files")
    print("   files.download('outputs.zip')")
    print("=" * 70)

if __name__ == "__main__":
    # 사용 예시
    print("사용 방법:")
    print("1. GitHub에서 클론:")
    print("   quick_start(method='github', github_url='https://github.com/USER/REPO.git')")
    print("\n2. Google Drive에서 가져오기:")
    print("   quick_start(method='drive')")
    print("\n3. GitHub + Drive 연동:")
    print("   quick_start(method='github', github_url='...', use_drive=True)")

