import cv2
import requests
import base64
import threading
import time
import textwrap
import numpy as np

# --- 설정 ---
API_URL = "http://127.0.0.1:8888/v1/chat/completions"
MODEL_DISPLAY_NAME = "Qwen3-VL-2B"

# [설정] AI 입력 해상도 (작을수록 빠름)
INFERENCE_W, INFERENCE_H = 320, 240 
# [설정] 화면 표시 배율 (2배 확대)
DISPLAY_SCALE = 2 

# 전역 변수
latest_result = "Ready..."
is_processing = False
last_latency = 0.00

def frame_to_base64(frame):
    resized = cv2.resize(frame, (INFERENCE_W, INFERENCE_H))
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return base64.b64encode(buffer).decode('utf-8')

def request_inference(b64_img):
    global latest_result, is_processing, last_latency
    start_t = time.time()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "In 5 words, Describe the scene."}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }
    ]

    payload = {
        "model": "Qwen3-VL-2B",
        "messages": messages,
        "max_tokens": 15,
        "temperature": 0.1
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=3)
        if response.status_code == 200:
            latest_result = response.json()['choices'][0]['message']['content'].strip()
            last_latency = time.time() - start_t
        else:
            latest_result = f"Err: {response.status_code}"
    except:
        latest_result = "Timeout"
    
    is_processing = False

def draw_ui(frame, result_text, latency):
    # 1. 화면 확대 (640x480)
    display = cv2.resize(frame, (INFERENCE_W * DISPLAY_SCALE, INFERENCE_H * DISPLAY_SCALE))
    
    # 2. 하단에 텍스트를 위한 검은 여백 추가 (높이 120픽셀)
    # copyMakeBorder(이미지, 위, 아래, 왼쪽, 오른쪽, 타입, 색상)
    text_area_height = 120
    canvas = cv2.copyMakeBorder(display, 0, text_area_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 3. 상단 정보 (작은 폰트)
    # 반투명 배경 (헤더)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"Model: {MODEL_DISPLAY_NAME}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Latency: {latency:.2f}s", (canvas.shape[1]-130, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # 4. 하단 결과 텍스트 (줄바꿈 처리)
    # 폰트 설정: 작게(0.55), 흰색
    font_scale = 0.55
    font_thickness = 1
    line_spacing = 25
    
    # 텍스트 시작 위치 (영상 바로 아래)
    text_y = display.shape[0] + 30 
    
    # 자동 줄바꿈 (화면 폭에 맞춰 자르기)
    wrap_width = 55 # 글자 수 기준 (폰트 크기에 따라 조절 필요)
    wrapped_lines = textwrap.wrap(result_text, width=wrap_width)

    cv2.putText(canvas, "", (1, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

    for i, line in enumerate(wrapped_lines):
        # 텍스트 영역을 벗어나지 않도록 최대 줄 수 제한 (옵션)
        if i > 3: break 
        cv2.putText(canvas, line, (70, text_y + (i * line_spacing)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return canvas

# --- 메인 ---
cap = cv2.VideoCapture(0)

# 카메라 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("--- Display Mode: Split View (Video Top / Text Bottom) ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # 추론 (비동기)
    if not is_processing:
        is_processing = True
        img_b64 = frame_to_base64(frame)
        threading.Thread(target=request_inference, args=(img_b64,), daemon=True).start()

    # UI 그리기 (영상 하단에 텍스트 박스 붙인 이미지 리턴)
    final_frame = draw_ui(frame, latest_result, last_latency)

    cv2.imshow('Smart View - Split Mode', final_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()