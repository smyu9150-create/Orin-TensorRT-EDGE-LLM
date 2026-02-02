import cv2
import requests
import base64
import threading
import time
import subprocess
import os
import sys
import signal
from collections import deque

# ==========================================
# [설정] 서버 및 경로
# ==========================================
WORK_DIR = os.path.expanduser("~/RMinte-Orin-TensorRT-EDGE-LLM")

SERVER_CMD = [
    "./build/examples/server/llm_server",
    "--engineDir", "./engines/qwen3-vl-2b-int4",                  
    "--multimodalEngineDir", "./visual_engines/qwen3-vl-2b-int4", 
    "--modelName", "Qwen3-VL-2B",
    "--port", "8888"
]

API_URL = "http://127.0.0.1:8888/v1/chat/completions"
MODEL_DISPLAY_NAME = "Qwen3-VL-2B"

# ==========================================
# [설정] 비전 로직
# ==========================================
WINDOW_SIZE = 4
STRIDE = 2

http_session = requests.Session()

# 전역 변수
latest_result = "Initializing Server..."
is_processing = False
last_latency = 0.0
frame_buffer = deque(maxlen=WINDOW_SIZE)
new_frame_count = 0 
last_capture_time = time.time()
skipped_triggers = 0  # [중요] 처리 중이라 누락된 횟수

def start_server():
    print(f"🚀 Starting LLM Server in {WORK_DIR}...")
    process = subprocess.Popen(
        SERVER_CMD, cwd=WORK_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid 
    )
    
    print("⏳ Waiting for TensorRT Engine to load...")
    server_ready = False
    while not server_ready:
        if process.poll() is not None:
            sys.exit(1)
        try:
            http_session.get("http://127.0.0.1:8888/health", timeout=1)
        except:
            time.sleep(2)
            print(".", end="", flush=True)
            continue
        server_ready = True
        print("\n✅ Server is Ready!")

    def log_reader(proc):
        for line in proc.stdout: pass
    threading.Thread(target=log_reader, args=(process,), daemon=True).start()
    return process

def frame_to_base64(frame):
    resized = cv2.resize(frame, (320, 240)) 
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return base64.b64encode(buffer).decode('utf-8')

def request_inference(b64_list):
    global latest_result, is_processing, last_latency
    start_t = time.time()
    
    content_list = [{"type": "text", "text": "Describe action in short."}]
    for b64_img in b64_list:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
    
    payload = {
        "model": "Qwen3-VL-2B",
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 5,
        "temperature": 0.1
    }
    
    try:
        response = http_session.post(API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            latest_result = response.json()['choices'][0]['message']['content'].strip()
            last_latency = time.time() - start_t
        else:
            latest_result = f"Error: {response.status_code}"
    except:
        latest_result = "Conn Timeout"
    
    # 처리가 다 끝난 후에 플래그 해제
    is_processing = False

if __name__ == "__main__":
    server_process = start_server()
    latest_result = "Analyzing..."

    cap = cv2.VideoCapture(0)
    print(f"--- {MODEL_DISPLAY_NAME} Monitor Mode ---")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # 0.25초마다 캡처
            if current_time - last_capture_time > 0.125:
                last_capture_time = current_time
                img_b64 = frame_to_base64(frame)
                frame_buffer.append(img_b64)
                new_frame_count += 1

                # [트리거 조건] 버퍼가 찼고, 스트라이드만큼 새로운 프레임이 들어왔을 때
                if len(frame_buffer) == WINDOW_SIZE and new_frame_count >= STRIDE:
                    if not is_processing:
                        # [상태: 처리 가능] -> 추론 시작
                        is_processing = True
                        new_frame_count = 0 # 카운터 리셋
                        threading.Thread(target=request_inference, args=(list(frame_buffer),), daemon=True).start()
                    else:
                        # [상태: 처리 중(Busy)] -> 요청 스킵(Drop)
                        skipped_triggers += 1
                        new_frame_count = 0 # 중요: 기회를 날렸으므로 카운터는 리셋 (다음 기회를 노림)

            # --- UI 그리기 ---
            overlay = frame.copy()
            # 상단 검은 바 영역 확장 (2줄 쓰기 위해 60px)
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # [1줄] 모델명 & 레이턴시
            cv2.putText(frame, f"Model: {MODEL_DISPLAY_NAME}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Lat: {last_latency:.2f}s", (frame.shape[1]-120, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

            # [2줄] 버퍼 / 스트라이드 / 스킵 정보 (핵심)
            # Buf: 4/4 | Next: 1/1 | Skip: 12
            # 색상: Skip이 발생하면 빨간색으로 표시하기 위해 조건문 사용
            status_color = (0, 255, 255) # 노란색 (기본)
            if is_processing: status_color = (0, 0, 255) # 처리 중이면 빨간색 느낌
            
            status_text = f"Buf: {len(frame_buffer)}/{WINDOW_SIZE} | Next: {new_frame_count}/{STRIDE} | Skip: {skipped_triggers}"
            cv2.putText(frame, status_text, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # [3줄] 결과 텍스트 (하단에 배치하거나 오버레이 하단에)
            cv2.putText(frame, f"> {latest_result}", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('Jetson Orin - Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("\n🛑 Shutting down...")
        http_session.close()
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        cap.release()
        cv2.destroyAllWindows()