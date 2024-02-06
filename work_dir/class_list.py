import json

file_path = "./class_list.txt"  # 텍스트 파일 경로

# 텍스트 파일을 읽어들여 리스트로 저장
with open(file_path, 'r') as file:
    lines = file.readlines()

# 텍스트를 원소 리스트로 변환
data_list = [line.strip() for line in lines]

# 원소가 5번 이상 중복되는 경우 해당 원소가 나타나는 순서의 번호와 원소 이름 출력
from collections import defaultdict

element_count = defaultdict(list)
for idx, element in enumerate(data_list):
    element_count[element].append(idx)

result = []
for element, indices in element_count.items():
    if len(indices) >= 5:
        start_index = indices[0]  # 첫 번째로 등장하는 인덱스
        start_time = start_index  # 초 단위로 변환할 경우
        result.append([element, f"{start_time} 초"])

result.append(['end', f"{len(data_list)}초"])

# ocr_to_tts.py에서 계산했던 티 샷의 비거리 추가
with open('shot_distance.txt', 'r') as file:
    txt_data = file.read().split()

result[0].extend(txt_data[:2])

file_path = "./class_result.json"
# result 리스트를 JSON 파일에 저장
with open(file_path, 'w') as file:
    json.dump(result, file)

