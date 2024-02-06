import os
import base64
import json

# WAV 파일이 있는 디렉토리 경로
wav_directory = './output_tts/'

# JSON 파일을 저장할 디렉토리 경로
output_json_directory = './output_json/'

# output_json 디렉토리가 없으면 생성
if not os.path.exists(output_json_directory):
    os.makedirs(output_json_directory)

# WAV 파일을 JSON 형식으로 변환하고 output_json 폴더로 이동
for filename in os.listdir(wav_directory):
    if filename.endswith('.wav'):
        # WAV 파일 읽기
        wav_file_path = os.path.join(wav_directory, filename)
        with open(wav_file_path, 'rb') as wav_file:
            wav_content = wav_file.read()

        # WAV 파일을 base64로 인코딩하여 JSON 형식으로 변환
        audio_base64 = base64.b64encode(wav_content).decode('utf-8')
        json_data = {'audio_base64': audio_base64}

        # JSON 파일로 변환된 데이터를 저장할 경로 설정
        output_json_path = os.path.join(output_json_directory, os.path.splitext(filename)[0] + '_tts.json')

        # JSON 파일로 저장
        with open(output_json_path, 'w') as json_file:
            json.dump(json_data, json_file)

        # 변환된 JSON 파일 생성 확인
        print(f"Created: {output_json_path}")
