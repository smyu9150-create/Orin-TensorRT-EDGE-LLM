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
VIDEO_DIR = os.path.expanduser("~/data/normal_350") 

CLASSES = ["abuse", "fighting", "road accident", "robbery", "shooting", "normal"]

# UI 색상 정의 (BGR)
CLASS_COLORS = {
    "normal": (0, 255, 0),       # Green
    "abuse": (0, 165, 255),      # Orange
    "fighting": (0, 0, 255),     # Red
    "road accident": (0, 255, 255), # Yellow
    "robbery": (255, 0, 255),    # Magenta
    "shooting": (0, 0, 139),     # Dark Red
    "initializing": (100, 100, 100), 
    "api error": (50, 50, 200)   
}

WINDOW_SIZE = 4 
STRIDE = 3
SAMPLE_INTERVAL = 0.25 

# --- [전역 변수] ---
latest_ui_label = "INITIALIZING"
latest_ui_color = CLASS_COLORS["initializing"]
is_inferencing = False
video_segment_predictions = [] 

# --- [1. 헬퍼 함수] ---
def normalize_label(text):
    clean = str(text).lower().replace("_", " ").strip()
    if "road" in clean or "accident" in clean or "crash" in clean: return "road accident"
    if "shoot" in clean or "gun" in clean or "fire" in clean: return "shooting"
    if "rob" in clean or "steal" in clean: return "robbery"
    if "fight" in clean or "brawl" in clean: return "fighting"
    if "abu" in clean or "hit" in clean or "slap" in clean: return "abuse"
    return "normal"

# [복구됨] 파일명에서 구체적인 정답 클래스 추출
def get_ground_truth_from_filename(filename):
    fname_clean = filename.lower().replace(" ", "")
    for cls in CLASSES:
        # normal은 다른 단어에 포함될 수도 있으니(예: abnormal) 주의하지만,
        # 여기서는 단순히 포함 여부 확인
        if cls.replace(" ", "") in fname_clean:
            return cls
    return "normal"

def frame_to_base64(frame):
    resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')

def draw_ui(frame, filename, label_text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(frame, f"File: {filename[:25]}...", (10, 28), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    disp_text = label_text.upper()
    text_size = cv2.getTextSize(disp_text, font, 1.2, 3)[0]
    cv2.rectangle(frame, (10, 50), (10 + text_size[0] + 20, 50 + text_size[1] + 20), (255, 255, 255), -1) 
    cv2.putText(frame, disp_text, (20, 95), font, 1.2, color, 3, cv2.LINE_AA)

    if label_text != "normal" and label_text not in ["initializing", "api error"]:
        cv2.rectangle(frame, (0, 0), (w, h), color, 8)

def get_video_files(directory):
    if not os.path.exists(directory):
        return []
    extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)

# --- [2. AI 추론 쓰레드] ---
def inference_worker(b64_buffer):
    global latest_ui_label, latest_ui_color, is_inferencing, video_segment_predictions
    
    prompt_text = (
        "Role: You are a Professional CCTV Safety AI. Identify the event based on CLEAR VISUAL EVIDENCE.\n"
        "Task: Select exactly ONE category that best describes the dominant event in the video.\n\n"
        "1. Shooting\n"
        "   - Keywords: Firearm (Gun/Pistol/Rifle), Shooting stance, Muzzle flash, Aiming at person.\n"
        "2. Robbery\n"
        "   - Keywords: Snatching bag/item, Forced entry, Breaking glass, Threatening staff/cashier.\n"
        "3. RoadAccident\n"
        "   - Keywords: Vehicle collision (Car/Bus/Truck), Motorcycle/Bike fall, Vehicle hitting pedestrian.\n"
        "4. Fighting\n"
        "   - Keywords: Physical brawl, Mutual punching/kicking, Wrestling, Multiple people tangled (No weapons).\n"
        "5. Abuse\n"
        "   - Keywords: One-sided assault, Pushing, Slapping, Physical harassment (No weapons).\n"
        "6. Normal\n"
        "   - Keywords: Routine walking, Standing, Jogging, Regular traffic, Static scenes, Safe interactions.\n\n"
        "Output Requirement: Output ONLY the category name without any extra text or punctuation."
    )

    content_list = [{"type": "text", "text": prompt_text}]
    for b64_img in b64_buffer:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

    payload = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 5, 
        "temperature": 0.0 
    }

    try:
        start_t = time.time()
        response = requests.post(API_URL, json=payload, timeout=8)
        
        if response.status_code == 200:
            raw = response.json()['choices'][0]['message']['content'].strip()
            norm = normalize_label(raw)
            print(f"  [AI] {norm.upper()} ({time.time() - start_t:.2f}s)")
            
            latest_ui_label = norm
            latest_ui_color = CLASS_COLORS.get(norm, (100, 100, 100))
            video_segment_predictions.append(norm)
        else:
            latest_ui_label = "API ERROR"
            latest_ui_color = CLASS_COLORS["api error"]

    except Exception:
        latest_ui_label = "TIMEOUT"
        latest_ui_color = CLASS_COLORS["api error"]
    
    finally:
        is_inferencing = False

# --- [3. 결과 처리 (핵심 변경 부분)] ---

def decide_final_verdict(predictions):
    """
    1. 연속성/밀도 조건 (기존 로직 유지)
    2. 유니온 조건 (추가됨: 비정상 총합이 과반수일 때)
    """
    if not predictions: return "normal"
    
    n = len(predictions)
    
    # --- [기존 로직 1] 데이터가 너무 적을 때 (2개 미만) ---
    if n < 2:
        for p in predictions:
            if p != "normal":
                return p
        return "normal"

    detected_candidates = []

    # --- [기존 로직 2] 연속성 검사 (Consecutive Check) ---
    for i in range(n - 1):
        curr = predictions[i]
        nxt = predictions[i+1]
        
        # 'normal'이 아니면서, 앞뒤가 똑같은 클래스일 때
        if curr != "normal" and curr == nxt:
            detected_candidates.append(curr)

    # --- [기존 로직 3] 밀도 검사 (Density Check) ---
    if n >= 10:
        for i in range(n - 9):
            window = predictions[i:i+10]
            counts = Counter(window)
            
            for cls, count in counts.items():
                if cls != "normal" and count >= 3:
                    detected_candidates.append(cls)

    # --- [판정 1] 기존 엄격한 규칙(연속/밀도)에 걸린 게 있다면 그것을 우선 반환 ---
    if detected_candidates:
        return Counter(detected_candidates).most_common(1)[0][0]

    # --- [판정 2 (신규 추가)] 유니온(Union) / 과반수 검사 ---
    # 위 엄격한 규칙에 안 걸렸더라도, 전체 예측 중 '비정상'의 합계가 '정상'보다 많으면 비정상 처리
    counts = Counter(predictions)
    normal_count = counts["normal"]
    total_abnormal_count = n - normal_count  # 전체 - 정상 = 비정상 개수

    # 비정상 예측이 전체의 50%를 초과하면
    if total_abnormal_count > normal_count:
        # normal을 제외하고 가장 많이 등장한 클래스를 범인으로 지목
        del counts["normal"]
        if counts:
            return counts.most_common(1)[0][0]

    # 여기까지 왔으면 진짜 정상
    return "normal"

def print_final_report(filenames, y_true, y_pred):
    print("\n" + "="*80)
    print(f"📄 MULTI-CLASS RESULT REPORT ({len(filenames)} Videos)")
    print("="*80)
    print(f"{'FILENAME':<25} | {'PRED':<15} | {'TRUE':<15} | {'RES'}")
    print("-" * 80)
    correct = 0
    for f, p, t in zip(filenames, y_pred, y_true):
        disp_name = f if len(f) < 22 else f[:10] + ".." + f[-10:]
        mark = "✅" if p == t else "❌"
        if p == t: correct += 1
        print(f"{disp_name:<25} | {p:<15} | {t:<15} | {mark}")
    
    acc_str = f"{correct/len(filenames)*100:.1f}%" if len(filenames) > 0 else "N/A"
    print(f"\n📊 Accuracy: {acc_str}")

def plot_confusion_matrix(y_true, y_pred):
    if not y_true: return
    # 모든 클래스에 대한 매트릭스 출력
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    acc = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    labels = [c.title() for c in CLASSES]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.show()

# --- [4. 메인 실행] ---
if __name__ == "__main__":
    video_files = get_video_files(VIDEO_DIR)
    
    final_data = {'names': [], 'true': [], 'pred': []}

    print(f"▶ Start processing {len(video_files)} videos (Multi-Class Logic)...")

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        filename = os.path.basename(video_path)
        
        # [복구됨] 파일명에서 구체적인 정답(fighting, robbery 등) 추출
        gt_label = get_ground_truth_from_filename(filename)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        wait_ms = int(1000 / fps) 

        frame_buffer = deque(maxlen=WINDOW_SIZE)
        video_segment_predictions = []
        latest_ui_label = "SCANNING..."
        latest_ui_color = CLASS_COLORS["initializing"]
        
        last_sample_time = time.time()
        sample_count = 0
        
        print(f"\n Playing: {filename} (GT: {gt_label})")

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_disp = cv2.resize(frame, (960, 540))
            
            if time.time() - last_sample_time > SAMPLE_INTERVAL:
                last_sample_time = time.time()
                frame_buffer.append(frame_to_base64(frame))
                sample_count += 1
                
                if len(frame_buffer) == WINDOW_SIZE and sample_count >= STRIDE and not is_inferencing:
                    is_inferencing = True
                    sample_count = 0
                    threading.Thread(target=inference_worker, args=(list(frame_buffer),), daemon=True).start()

            draw_ui(frame_disp, filename, latest_ui_label, latest_ui_color)
            cv2.imshow('AI Monitor', frame_disp)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print_final_report(final_data['names'], final_data['true'], final_data['pred'])
                plot_confusion_matrix(final_data['true'], final_data['pred'])
                exit()
            elif key == ord('n'):
                break

        cap.release()
        
        # [변경됨] 개별 클래스에 대해 연속성/밀도를 검사하는 함수 호출
        verdict = decide_final_verdict(video_segment_predictions)
        
        final_data['names'].append(filename)
        final_data['true'].append(gt_label)
        final_data['pred'].append(verdict)
        print(f" Final Decision: {verdict.upper()}")

    cv2.destroyAllWindows()
    print_final_report(final_data['names'], final_data['true'], final_data['pred'])
    plot_confusion_matrix(final_data['true'], final_data['pred'])