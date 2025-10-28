#!/usr/bin/env python3
"""
학습된 SAC 모델을 ROS2 추론용으로 변환하는 스크립트.

문제: configure_learning_rates()로 커스텀한 optimizer 구조 때문에
     Stable Baselines3의 SAC.load()가 실패합니다.

해결: policy weights만 추출하여 새로운 SAC 모델로 저장합니다.
"""

import sys
import zipfile
import tempfile
import shutil
from pathlib import Path


def convert_model_for_inference(input_path: str, output_path: str):
    """
    학습된 SAC 모델을 추론용 모델로 변환합니다.

    Args:
        input_path: 원본 모델 경로 (.zip)
        output_path: 변환된 모델 경로 (.zip)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"❌ 입력 파일이 존재하지 않습니다: {input_path}")
        return False

    print(f"🔄 모델 변환 시작...")
    print(f"  입력: {input_path}")
    print(f"  출력: {output_path}")

    try:
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            extract_dir = tmpdir_path / "extracted"
            new_model_dir = tmpdir_path / "new_model"
            extract_dir.mkdir()
            new_model_dir.mkdir()

            # 원본 모델 압축 해제
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"\n📦 원본 모델 내용:")
            for file in sorted(extract_dir.iterdir()):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name:30s} ({size_mb:.2f} MB)")

            # 필요한 파일만 복사
            required_files = [
                'data',
                'policy.pth',
                'pytorch_variables.pth',
                '_stable_baselines3_version',
                'system_info.txt'
            ]

            print(f"\n✅ 추론용 모델 생성 (optimizer 제외):")
            for filename in required_files:
                src = extract_dir / filename
                if src.exists():
                    dst = new_model_dir / filename
                    shutil.copy2(src, dst)
                    size_mb = dst.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {filename:30s} ({size_mb:.2f} MB)")
                else:
                    print(f"  ⚠ {filename:30s} (없음)")

            # 새로운 zip 파일 생성
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for file_path in new_model_dir.iterdir():
                    zip_out.write(file_path, file_path.name)

            print(f"\n✅ 변환 완료!")
            print(f"  저장 위치: {output_path}")
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            input_size_mb = input_path.stat().st_size / (1024 * 1024)
            print(f"  파일 크기: {output_size_mb:.2f} MB (원본: {input_size_mb:.2f} MB)")
            print(f"  크기 감소: {input_size_mb - output_size_mb:.2f} MB")

            return True

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 convert_model_for_inference.py <input_model.zip> [output_model.zip]")
        print()
        print("예시:")
        print("  python3 convert_model_for_inference.py train_sac_cnn/sac-latest.zip models/sac-inference.zip")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # 자동으로 출력 경로 생성
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}-inference.zip")

    success = convert_model_for_inference(input_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
