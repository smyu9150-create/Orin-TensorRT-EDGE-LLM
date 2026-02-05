import cv2
import requests
import base64
import threading
import time
import os
import glob
from collections import deque, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# --- [설정 영역] ---
API_URL = "http://127.0.0.1:8888/v1/chat/completions"
VIDEO_DIR = os.path.expanduser("~/data/Original_Videos4sam")
CLASSES = ["abuse", "fighting", "road accident", "robbery", "shooting", "normal"]

CLASS_COLORS = {
    "normal": (0, 255, 0), "abuse": (0, 165, 255), "fighting": (0, 0, 255),
    "road accident": (0, 255, 255), "robbery": (255, 0, 255), "shooting": (0, 0, 139),
    "initializing": (100, 100, 100), "api error": (50, 50, 200)   
}

WINDOW_SIZE = 4 
STRIDE = 3
SAMPLE_INTERVAL = 0.25 

# --- [전역 변수] ---
latest_ui_label = "INITIALIZING"
latest_ui_color = CLASS_COLORS["initializing"]
is_inferencing = False
video_segment_predictions = []  
timeline_data = []              
inference_triggers = [] 

# --- [1. 헬퍼 함수] ---
def normalize_label(text):
    clean = str(text).lower().replace("_", " ").strip()
    if any(x in clean for x in ["road", "accident", "crash"]): return "road accident"
    if any(x in clean for x in ["shoot", "gun", "fire"]): return "shooting"
    if any(x in clean for x in ["rob", "steal"]): return "robbery"
    if any(x in clean for x in ["fight", "brawl"]): return "fighting"
    if any(x in clean for x in ["abu", "hit", "slap"]): return "abuse"
    return "normal"

def get_ground_truth_from_path(video_path):
    """폴더명을 기반으로 정답(GT)을 추출합니다."""
    path_lower = video_path.lower()
    parent_dir = os.path.basename(os.path.dirname(video_path)).lower()
    if "abuse" in parent_dir: return "abuse"
    if "fight" in parent_dir: return "fighting"
    if "road" in parent_dir or "accident" in parent_dir: return "road accident"
    if "robbery" in parent_dir: return "robbery"
    if "shoot" in parent_dir: return "shooting"
    return "normal"

def frame_to_base64(frame):
    resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')

# --- [그리기 함수들] ---
def draw_timeline(frame, current_frame, total_frames, history, triggers, fps):
    h, w = frame.shape[:2]
    bar_height = 30
    y_start = h - bar_height
    if total_frames <= 0 or fps <= 0: return
    progress_ratio = min(1.0, max(0.0, current_frame / total_frames))
    cv2.rectangle(frame, (0, y_start), (w, h), (40, 40, 40), -1)
    cur_x = int(progress_ratio * w)
    if progress_ratio >= 1.0: cur_x = w
    start_x = 0
    for item in history:
        end_x = int((item['frame'] / total_frames) * w)
        draw_end_x = min(end_x, cur_x)
        color = CLASS_COLORS.get(item['label'], (100, 100, 100))
        if draw_end_x > start_x:
            cv2.rectangle(frame, (start_x, y_start), (draw_end_x, h), color, -1)
        start_x = end_x
    if start_x < cur_x:
        cv2.rectangle(frame, (start_x, y_start), (cur_x, h), (100, 100, 100), -1)
    for trig_frame in triggers:
        trig_x = int((trig_frame / total_frames) * w)
        cv2.line(frame, (trig_x, y_start), (trig_x, h), (255, 255, 255), 1)

def draw_ui(frame, filename, label_text, color):
    h, w = frame.shape[:2]
    if label_text not in ["normal", "initializing", "api error"]:
        cv2.rectangle(frame, (0, 0), (w, h - 30), color, 10)
    cv2.putText(frame, label_text.upper(), (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
    cv2.putText(frame, label_text.upper(), (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, filename[:30], (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def get_video_files(directory):
    files = []
    for ext in ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm'):
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)

# --- [2. AI 추론 쓰레드] ---
def inference_worker(b64_buffer, capture_frame_idx):
    global latest_ui_label, latest_ui_color, is_inferencing, video_segment_predictions, timeline_data
    prompt_text = (
        "Role: Professional CCTV AI.\n"
        "Task: Select ONE from {Shooting, Robbery, RoadAccident, Fighting, Abuse, Normal}. Output ONLY the word."
    )    
    content_list = [{"type": "text", "text": prompt_text}]
    for b64_img in b64_buffer:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
    payload = {"model": "Qwen/Qwen3-VL-2B-Instruct", "messages": [{"role": "user", "content": content_list}], "max_tokens": 10, "temperature": 0.0}
    try:
        response = requests.post(API_URL, json=payload, timeout=8)
        if response.status_code == 200:
            norm = normalize_label(response.json()['choices'][0]['message']['content'])
            latest_ui_label, latest_ui_color = norm, CLASS_COLORS.get(norm, (100, 100, 100))
            video_segment_predictions.append(norm)
            timeline_data.append({'frame': capture_frame_idx, 'label': norm})
    except: latest_ui_label = "API ERROR"
    finally: is_inferencing = False

# --- [3. 결과 처리] ---
def decide_final_verdict(predictions):
    if not predictions: return "normal"
    counts = Counter(predictions)
    others = {k: v for k, v in counts.items() if k != "normal"}
    return max(others, key=others.get) if others else "normal"

# --- [4. 메인 실행] ---
if __name__ == "__main__":
    video_files = get_video_files(VIDEO_DIR)
    final_data = {'names': [], 'true': [], 'pred': []}

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue
        
        filename = os.path.basename(video_path)
        gt_label = get_ground_truth_from_path(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        frame_buffer = deque(maxlen=WINDOW_SIZE)
        video_segment_predictions, timeline_data, inference_triggers = [], [], []
        latest_ui_label, latest_ui_color = "SCANNING", (100, 100, 100)
        last_sample_time, sample_count = 0, 0

        print(f"\n Playing: {filename} (GT: {gt_label})")

        while True:
            ret, frame = cap.read()
            if not ret: break
            cur_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_f > total_frames: total_frames = cur_f

            if time.time() - last_sample_time > SAMPLE_INTERVAL:
                last_sample_time = time.time()
                frame_buffer.append(frame_to_base64(frame))
                sample_count += 1
                if len(frame_buffer) == WINDOW_SIZE and sample_count >= STRIDE and not is_inferencing:
                    is_inferencing = True
                    sample_count = 0
                    inference_triggers.append(cur_f)
                    threading.Thread(target=inference_worker, args=(list(frame_buffer), cur_f), daemon=True).start()

            frame_disp = cv2.resize(frame, (960, 540))
            draw_ui(frame_disp, filename, latest_ui_label, latest_ui_color)
            draw_timeline(frame_disp, cur_f, total_frames, timeline_data, inference_triggers, fps)
            cv2.imshow('Qwen3-VL-2B_muti_class', frame_disp)

            # [핵심 수정] waitKey(1)로 변경하여 빠른 분석 수행
            if cv2.waitKey(1) & 0xFF == ord('q'): exit()

        cap.release()
        while is_inferencing: time.sleep(0.01) # Flush 대기

        verdict = decide_final_verdict(video_segment_predictions)
        final_data['names'].append(filename)
        final_data['true'].append(gt_label)
        final_data['pred'].append(verdict)
        print(f" Final Decision: {verdict.upper()}")

    cv2.destroyAllWindows()
    # 최종 리포트 및 매트릭스 출력
    cm = confusion_matrix(final_data['true'], final_data['pred'], labels=CLASSES)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Accuracy: {accuracy_score(final_data['true'], final_data['pred'])*100:.1f}%"); plt.show()