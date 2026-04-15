from typing import Dict, List

# 사용할 피처(열) 이름
ANGLE_COLUMNS: List[str] = ["hip_angle", "knee_angle", "ankle_angle"]
LABEL_COLUMN: str = "label"

# 라벨 번호 -> 라벨 이름
LABEL_NAME_MAPPING: Dict[int, str] = {
    0: "Normal",
    1: "Incomplete injury",
    2: "Complete injury",
    3: "Implanted_no sti",
    4: "Implanted_sti",
}

# 0,1,2에 대한 이름 리스트 (순서 중요)
BASE_LABEL_NAMES = [
    LABEL_NAME_MAPPING[0],
    LABEL_NAME_MAPPING[1],
    LABEL_NAME_MAPPING[2],
]
