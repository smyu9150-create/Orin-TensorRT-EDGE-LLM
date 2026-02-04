import cv2
import requests
import base64
import threading
import time
import os
import glob
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# --- [설정] ---
API_URL = "http://127.0.0.1:8888/v1/chat/completions"
VIDEO_DIR = os.path.expanduser("~/data/video") 
LABELS = ["ABNORMAL", "NORMAL"]
WINDOW_SIZE = 4
STRIDE = 1

# --- [전역 변수] ---
latest_label = "INITIALIZING"
latest_color = (100, 100, 100) 
is_processing = False
last_latency = 0.0
frame_buffer = deque(maxlen=WINDOW_SIZE)
new_frame_count = 0 
last_capture_time = 0

current_video_stats = [] 
all_results = [] 

def frame_to_base64(frame):
    resized = cv2.resize(frame, (320, 320)) 
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

def request_inference(b64_list):
    global latest_label, latest_color, is_processing, last_latency, current_video_stats
    start_t = time.time()
    
    system_instruction = (
    "Role: You are a strict CCTV Safety AI. Your goal is to detect PROVEN anomalies only.\n"
    "Task: Classify the video segment as 'Normal' or 'Abnormal'.\n\n"
    "[Definitions]\n"
    "1. Abnormal (CRITICAL & CLEAR Events Only):\n"
    "   - Violence: Fighting, Punching, Kicking, Shooting, Assault.\n"
    "   - Group Violence: Multiple people tangled, wrestling, or aggressive group brawl.\n"
    "   - Sudden Attack: Fast swinging arm (punching), sudden lunge at a person.\n"
    "   - Crime: Robbery, Burglary, Shoplifting, Vandalism, Arson.\n"
    "   - Disaster: Explosion, Car Accident, Fire.\n"
    "   * NOTE: The event must be visually clear and happening right now.\n\n"
    "2. Normal (Safe or Unclear Situations):\n"
    "   - Routine: Walking, Running (jogging/hurrying), Standing, Sitting, Crowds.\n"
    "   - Environment: Dark scenes, Blurry footage, Moving shadows.\n\n"
    "Output Requirement: Output ONLY the single word 'Normal' or 'Abnormal' without any punctuation."
    )

    content_list = [{"type": "text", "text": system_instruction}]
    for b64_img in b64_list:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

    payload = {
        "model": "Qwen3-VL-2B-Instruct", 
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 5, 
        "temperature": 0.0 
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        latency = time.time() - start_t
        
        if response.status_code == 200:
            raw_content = response.json()['choices'][0]['message']['content'].strip().upper()
            
            if "ABNORMAL" in raw_content or "DANGER" in raw_content or "FIGHT" in raw_content:
                latest_label = "ABNORMAL"
                latest_color = (0, 0, 255) 
                current_video_stats.append("ABNORMAL")
            elif "NORMAL" in raw_content:
                latest_label = "NORMAL"
                latest_color = (0, 255, 0) 
                current_video_stats.append("NORMAL")
            else:
                latest_label = "UNCERTAIN" 
                latest_color = (0, 255, 255) 
                current_video_stats.append("NORMAL") 

            last_latency = latency
            print(f"  -> [AI Segment] {latest_label} ({latency:.2f}s)")
            
        else:
            print(f"Err: {response.status_code}")
            latest_label = "API ERROR"
            latest_color = (0, 165, 255) 

    except Exception as e:
        print(f"Req Error: {e}")
        latest_label = "TIMEOUT"
        latest_color = (128, 128, 128)
    
    is_processing = False

def get_video_files(directory):
    if not os.path.exists(directory): return []
    extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)

def draw_ui(frame, filename, label, color, latency):
    h, w = frame.shape[:2]
    if label == "ABNORMAL":
        cv2.rectangle(frame, (0, 0), (w, h), color, 10)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{label}", (20, 50), font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"{label}", (20, 50), font, 1.0, color, 2, cv2.LINE_AA)
    
    info = f"{filename[:15]}.. | {latency:.2f}s"
    cv2.putText(frame, info, (20, 80), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, info, (20, 80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def get_ground_truth(filename):
    fname_lower = filename.lower()
    if "normal" in fname_lower:
        return "NORMAL"
    else:
        return "ABNORMAL"

def check_final_verdict(stats):
    n = len(stats)
    if n < 2: 
        return "ABNORMAL" if "ABNORMAL" in stats else "NORMAL"

    for i in range(n - 1):
        if stats[i] == "ABNORMAL" and stats[i+1] == "ABNORMAL":
            return "ABNORMAL"

    if n >= 5:
        for i in range(n - 4):
            window = stats[i:i+5]
            if window.count("ABNORMAL") >= 3:
                return "ABNORMAL"
    
    if stats.count("ABNORMAL") >= 3:
        return "ABNORMAL"

    return "NORMAL"

def plot_beautiful_matrix(results):
    if not results: return

    y_true = [r['true'] for r in results]
    y_pred = [r['pred'] for r in results]
    
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    plt.figure(figsize=(8, 7))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=LABELS, yticklabels=LABELS,
                     annot_kws={"size": 16}, cbar=False)
    plt.title(f"Confusion Matrix (Acc: {accuracy:.1%})", fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- [새로 추가된 함수] 결과 리포트 출력 ---
def print_final_summary(results):
    if not results: return
    
    print("\n" + "="*80)
    print(f"📄 FINAL PROCESSING REPORT ({len(results)} Videos)")
    print("="*80)
    print(f"{'FILENAME':<35} | {'PREDICT':<10} | {'TRUTH':<10} | {'RESULT'}")
    print("-" * 80)
    
    correct_cnt = 0
    for res in results:
        fname = res['filename']
        if len(fname) > 33: fname = fname[:30] + "..."
        
        pred = res['pred']
        gt = res['true']
        is_correct = (pred == gt)
        
        if is_correct: 
            correct_cnt += 1
            mark = "✅ OK"
        else:
            mark = "❌ FAIL"
            
        print(f"{fname:<35} | {pred:<10} | {gt:<10} | {mark}")
        
    acc = correct_cnt / len(results) * 100
    print("-" * 80)
    print(f"📊 Total Accuracy: {acc:.2f}% ({correct_cnt}/{len(results)})")
    print("="*80 + "\n")

if __name__ == "__main__":
    print(f"Searching in: {VIDEO_DIR}")
    video_files = get_video_files(VIDEO_DIR)
    
    if not video_files:
        print(f"❌ Error: No videos found in {VIDEO_DIR}")
        exit()

    print(f"📂 Found {len(video_files)} videos.")
    
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        filename = os.path.basename(video_path)
        true_label = get_ground_truth(filename)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        wait_ms = int(1000 / fps)

        frame_buffer.clear()
        current_video_stats = [] 
        latest_label = "SCANNING..."
        latest_color = (100, 100, 100)
        new_frame_count = 0
        last_capture_time = time.time()
        
        print(f"\n▶ Playing: {filename} (GT: {true_label})")

        while True:
            ret, frame = cap.read()
            if not ret: break 

            if frame.shape[1] < 640:
                frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

            curr_t = time.time()

            if curr_t - last_capture_time > 0.25:
                last_capture_time = curr_t
                img_b64 = frame_to_base64(frame)
                frame_buffer.append(img_b64)
                new_frame_count += 1

                if len(frame_buffer) == WINDOW_SIZE and new_frame_count >= STRIDE:
                    if not is_processing:
                        is_processing = True
                        new_frame_count = 0
                        threading.Thread(target=request_inference, args=(list(frame_buffer),), daemon=True).start()

            draw_ui(frame, filename, latest_label, latest_color, last_latency)
            cv2.imshow('UCF Crime Classifier', frame)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                print_final_summary(all_results) # 중간 종료 시에도 요약 출력
                plot_beautiful_matrix(all_results)
                exit()
            elif key == ord('n'):
                break 

        cap.release()
        
        final_verdict = check_final_verdict(current_video_stats)
        
        all_results.append({
            'filename': filename,
            'true': true_label,
            'pred': final_verdict
        })
        
        print(f"✅ Verdict: [{final_verdict}] (GT: {true_label})")

    cv2.destroyAllWindows()
    
    # [수정됨] 모든 처리 후 요약 리포트 출력 -> 그 다음 혼동 행렬
    print_final_summary(all_results)
    plot_beautiful_matrix(all_results)