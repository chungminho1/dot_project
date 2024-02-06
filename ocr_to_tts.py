import torch
import cv2
import easyocr
import os
import json
import wave
import base64
from gtts import gTTS

#선수 스코어가 음수인 경우 ~언더파로 수정
def replace_negative_numbers(text):
    if text.startswith('-') and text[1:].isdigit():
        return text[1:] + ' 언더파'
    return text

'''
OCR inference 시간 단축을 위해 지정된 특정 영역에서만 inference 수행하도록 boundary 설정, 5초마다 초기화하기
'''

def extract_text_from_video_frame_with_boundary(n, bounding_box):

    # 비디오 파일 경로
    video_path = './data_dot/testvideo1.mp4'

    reader = easyocr.Reader(['ko', 'en'], gpu=True)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    texts_list = []

    for i in range(0, total_frames, int(5*fps)):
        texts = []

        frame_number = int(i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if ret:
            # 바운더리 영역 내의 이미지 추출
            x1, y1, x2, y2 = bounding_box
            cropped_frame = frame[y1:y2, x1:x2]

            # 바운더리 내의 이미지에서 텍스트 추출
            result = reader.readtext(cropped_frame)
            for detection in result:
                detected_text = replace_negative_numbers(detection[1]) if detection[2] >= 0.9 else detection[1]
                texts.append(detected_text)

        else:
            print("Frame extraction failed.")

        # 5초 간격으로 추출된 OCR 결과를 리스트에 추가
        texts_list.append(texts)

    cap.release()

    return texts_list

#좌상단 현재 샷 정보
shot_information = extract_text_from_video_frame_with_boundary(5, (63, 42, 326, 100))

#좌상단 홀까지의 거리
to_hole = extract_text_from_video_frame_with_boundary(5, (331, 48, 396, 99))

#우하단 스코어보드
scoreboard = extract_text_from_video_frame_with_boundary(5, (1044, 506, 1181, 634))

shot_number = extract_text_from_video_frame_with_boundary(5, (61, 46, 103, 91))


#파일 전송을 위한 파일 포맷 변환
def wav_to_json(wav_file_path):
    # WAV 파일을 base64로 인코딩
    with open(wav_file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

    # JSON 데이터 생성
    json_data = {
        'audio_base64': audio_base64
    }

    # WAV 파일의 이름을 기반으로 JSON 파일 이름 생성
    json_output_filename = f'{os.path.splitext(os.path.basename(wav_file_path))[0]}.json'
    json_output_path = os.path.join(os.path.dirname(wav_file_path), json_output_filename)

    # JSON 파일에 저장
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data, json_file)

    return json_output_path

output_dir = './output_tts/'

print(shot_information[2])

#list 초기화
previous_text_list = None

#티샷 비거리
shot_distance = []

#현재 샷 정보
for i in range(len(shot_information)):
    text_list = shot_information[i]
    for text in text_list:
        if text.endswith('m') and text not in shot_distance:
            shot_distance.append(text)
    if not text_list:  # 리스트가 비어있는 경우 이전 리스트로 대체
        if previous_text_list:
            text_list = previous_text_list
        else:
            continue  # 첫 번째 리스트가 비어있는 경우, 처리하지 않고 넘어감

    text_to_speak = ' '.join(text_list)
    tts = gTTS(text=text_to_speak, lang='ko')
    file_name = f'shot_information{i}.wav'

    output_path = os.path.join(output_dir, file_name)
    tts.save(output_path)

previous_text_list = None

#티샷의 비거리 계산을 위한 shot_distance
for i in range(len(to_hole)):
    text_list = to_hole[i]
    for text in text_list:
        if text.endswith('m') and text not in shot_distance:
            shot_distance.append(text)
    if not text_list:  # 리스트가 비어있는 경우 이전 리스트로 대체
        if previous_text_list:
            text_list = previous_text_list
        else:
            continue  # 첫 번째 리스트가 비어있는 경우, 처리하지 않고 넘어감

    text_to_speak = ' '.join(text_list)
    tts = gTTS(text=text_to_speak, lang='ko')
    file_name = f'to_hole{i}.wav'

    output_path = os.path.join(output_dir, file_name)
    tts.save(output_path)

previous_text_list = None

#스코어보드
for i in range(len(scoreboard)):
    text_list = scoreboard[i]
    if not text_list:  # 리스트가 비어있는 경우 이전 리스트로 대체
        if previous_text_list:
            text_list = previous_text_list
        else:
            continue 
    # 첫 번째 리스트가 비어있는 경우, 처리하지 않고 넘어감

    text_to_speak = ' '.join(text_list)
    tts = gTTS(text=text_to_speak, lang='ko')
    file_name = f'scoreboard{i}.wav'

    output_path = os.path.join(output_dir, file_name)
    tts.save(output_path)

previous_text_list = None

#스코어보드 txt
for i in range(len(scoreboard)):
    text_list = scoreboard[i]
    if not text_list:
        if previous_text_list:
            text_list = previous_text_list
        else:
            continue
    json_data = json.dumps(scoreboard[i])
    directory = './txt/'
    file_name = f'scoreboard{i}.json'
    file_path = os.path.join(directory, file_name)
    with open(file_path,'w') as file:
        file.write(json_data)

previous_text_list = None

for i in range(len(shot_number)):
    text_list = shot_number[i]
    if not text_list:
        if previous_text_list:
            text_list = previous_text_list
        else:
            continue
    json_data = json.dumps(shot_number[i])
    directory = './txt/'
    file_name = f'shot_number{i}.json'
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
        file.write(json_data)    

print(shot_distance)
with open('./work_dir/shot_distance.txt', 'w') as file:
    for item in shot_distance:
        file.write(f"{item}\n")
