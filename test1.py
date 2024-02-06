import cv2
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import warnings
import time

warnings.filterwarnings("ignore", message="Failed to load image Python extension", category=UserWarning)

class_names = [
    "bunker",     
    "bunker_shot",  
    "hazard",   
    "just_human",  
    "nth_hole",   
    "on_green",  
    "putting_shot",   
    "rough",   
    "rough_and_green",   
    "rough_and_green_shot",   
    "rough_shot",  
    "sky",  
    "tee_shot"  
    ]

class_list = []

#학습된 checkpoint파일을 가져와서 프레임에 이미지 분류  적용(fps=1)
def process_video(video_path, model_path, fps=1):
    cap = cv2.VideoCapture(video_path)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=13)
    model.load_state_dict(torch.load('./work_dir/output_main.pt'), strict=False)
    model = model.to(device)
    model.eval()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)
    current_frame = 0
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_pil = Image.fromarray(frame)

        frame_tensor = transform(frame_pil)

        frame_tensor = frame_tensor.unsqueeze(0)
        frame_tensor = frame_tensor.to('cuda')

        with torch.no_grad():
            output = model(frame_tensor)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        current_time = current_frame / video_fps
        
        predicted_class = output.argmax().item()
        class_name = class_names[predicted_class]

        class_list.append(class_name)

        current_frame += interval

    cap.release()

video_path = './data_dot/testvideo1.mp4'
model_path = './work_dir/output_latest.pt'  
YOUR_NUM_CLASSES = 13

start_time=time.time()
#초 단위로 장면 분류된 프레임을 txt로 저장
process_video(video_path, model_path, fps=1)
print(class_list)
end_time=time.time()
print(f'{end_time-start_time}초')
