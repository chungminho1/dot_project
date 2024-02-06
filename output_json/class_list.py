file_path = '../work_dir/class_list.txt'
element_to_find = ''  # 찾고자 하는 원소

# 파일에서 원소를 읽어 리스트로 저장
with open(file_path, 'r') as file:
    lines = file.readlines()

# 리스트의 원소를 확인하면서 연속으로 등장하는 원소의 시작 인덱스 출력
count = 0
for idx, line in enumerate(lines):
    line = line.strip()  # 줄 바꿈 문자 제거
    if line == element_to_find:
        count += 1
        if count == 1:  # 새로운 연속이 시작될 때
            start_idx = idx
        if count >= 5:  # 5번 이상 연속 등장한 경우
            print(f"'{element_to_find}' started appearing at index: {start_idx}")
            count = 0  # count 초기화
    else:
        count = 0  # 연속이 끊긴 경우 count 초기화
