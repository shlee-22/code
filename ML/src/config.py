from pathlib import Path

# 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 기본 데이터 파일 경로
DATA_FILE_PATH = PROJECT_ROOT / "data" / "Extension_움직임데이터_3_read.xlsx"

# 학습에 사용할 라벨
TRAIN_LABELS = [0, 1, 2]

# Implanted 분석에 사용할 라벨
IMPLANTED_LABELS = [3, 4]

# Train/Test split 설정 (기존 유지)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Confusion matrix 계산용 CV 설정
N_SPLITS = 5
