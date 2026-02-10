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

# 실제 영상들이 들어있는 경로로 수정
VIDEO_DIR = os.path.expanduser("~/data/video")

# Updated CLASSES (14 categories)
CLASSES = [
    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "normal", "roadaccidents", "robbery", "shooting",
    "shoplifting", "stealing", "vandalism"
]

# Updated CLASS_COLORS with new categories
CLASS_COLORS = {
    "normal": (0, 255, 0),           # Green
    "abuse": (0, 165, 255),          # Orange
    "fighting": (0, 0, 255),         # Red
    "roadaccidents": (0, 255, 255),  # Yellow
    "robbery": (255, 0, 255),        # Magenta
    "shooting": (0, 0, 139),         # Dark Red
    "arrest": (255, 128, 0),         # Blue
    "arson": (0, 69, 255),           # Orange Red
    "assault": (147, 20, 255),       # Deep Pink
    "burglary": (130, 0, 75),        # Indigo
    "explosion": (0, 140, 255),      # Dark Orange
    "shoplifting": (203, 192, 255),  # Pink
    "stealing": (128, 0, 128),       # Purple
    "vandalism": (0, 215, 255),      # Gold
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
timeline_data = []              
inference_triggers = [] 

# --- [1. 헬퍼 함수] ---
def normalize_label(text):
    """Normalize AI output to match one of the 14 classes"""
    clean = str(text).lower().replace("_", " ").replace("-", " ").strip()
    
    # Direct matches first
    if clean in ["abuse", "arrest", "arson", "assault", "burglary", "explosion",
                 "fighting", "normal", "roadaccidents", "robbery", "shooting",
                 "shoplifting", "stealing", "vandalism"]:
        return clean
    
    # Keyword matching for variations
    if "road" in clean or "accident" in clean or "crash" in clean or "collision" in clean:
        return "roadaccidents"
    if "shoot" in clean or "gun" in clean or "fire" in clean or "firearm" in clean:
        return "shooting"
    if "rob" in clean or "robbery" in clean:
        return "robbery"
    if "fight" in clean or "brawl" in clean:
        return "fighting"
    if "abuse" in clean or "abusive" in clean:
        return "abuse"
    if "arrest" in clean or "detain" in clean:
        return "arrest"
    if "arson" in clean or "fire" in clean and "building" in clean:
        return "arson"
    if "assault" in clean or "attack" in clean:
        return "assault"
    if "burgl" in clean or "break" in clean and "enter" in clean:
        return "burglary"
    if "explo" in clean or "blast" in clean or "detonate" in clean:
        return "explosion"
    if "shoplift" in clean:
        return "shoplifting"
    if "steal" in clean or "theft" in clean:
        return "stealing"
    if "vandal" in clean or "damage" in clean or "destroy" in clean:
        return "vandalism"
    
    return "normal"

def get_ground_truth_from_filename(filename):
    """Extract ground truth label from filename"""
    fname_clean = filename.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    # Check each class (order matters - more specific first)
    if "roadaccident" in fname_clean or "accident" in fname_clean:
        return "roadaccidents"
    if "shoplifting" in fname_clean or "shoplift" in fname_clean:
        return "shoplifting"
    if "shooting" in fname_clean or "shoot" in fname_clean:
        return "shooting"
    if "robbery" in fname_clean or "rob" in fname_clean:
        return "robbery"
    if "fighting" in fname_clean or "fight" in fname_clean:
        return "fighting"
    if "arrest" in fname_clean:
        return "arrest"
    if "arson" in fname_clean:
        return "arson"
    if "assault" in fname_clean:
        return "assault"
    if "burglary" in fname_clean or "burglar" in fname_clean:
        return "burglary"
    if "explosion" in fname_clean or "explode" in fname_clean:
        return "explosion"
    if "stealing" in fname_clean or "steal" in fname_clean:
        return "stealing"
    if "vandalism" in fname_clean or "vandal" in fname_clean:
        return "vandalism"
    if "abuse" in fname_clean:
        return "abuse"
    
    return "normal"

def frame_to_base64(frame):
    resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')

# --- [타임라인 그리기 함수] ---
def draw_timeline(frame, current_frame, total_frames, history, triggers, fps):
    h, w = frame.shape[:2]
    bar_height = 30
    y_start = h - bar_height
    
    if total_frames <= 0 or fps <= 0: return

    # 진행률 계산 (0.0 ~ 1.0)
    progress_ratio = min(1.0, max(0.0, current_frame / total_frames))
    
    # 1. 배경 (전체)
    cv2.rectangle(frame, (0, y_start), (w, h), (40, 40, 40), -1)

    cur_x = int(progress_ratio * w)
    
    # [보정] 100% 진행시 1픽셀 오차 없이 끝까지 채움
    if progress_ratio >= 1.0:
        cur_x = w

    # 2. 과거 데이터 색칠
    start_x = 0
    for item in history:
        end_frame = item['frame']
        # 비율로 변환
        ratio = min(1.0, end_frame / total_frames)
        end_x = int(ratio * w)
        
        draw_end_x = min(end_x, cur_x)
        
        color = CLASS_COLORS.get(item['label'], (100, 100, 100))

        if draw_end_x > start_x:
            cv2.rectangle(frame, (start_x, y_start), (draw_end_x, h), color, -1)
        
        start_x = end_x
        if start_x >= cur_x: break
            
    # 3. 대기 중 구간 회색 처리
    if start_x < cur_x:
        waiting_color = CLASS_COLORS["initializing"]
        cv2.rectangle(frame, (start_x, y_start), (cur_x, h), waiting_color, -1)

    # 4. 인퍼런스 트리거 지점 (하얀색 작대기)
    for trig_frame in triggers:
        trig_ratio = min(1.0, trig_frame / total_frames)
        trig_x = int(trig_ratio * w)
        cv2.line(frame, (trig_x, y_start), (trig_x, h), (255, 255, 255), 1)

    # 5. 시간 눈금 (초 단위)
    total_seconds = total_frames / fps
    step_seconds = max(1, int(total_seconds / 6)) 
    font_color = (200, 200, 200)
    
    for t_sec in range(0, int(total_seconds) + 1, step_seconds):
        ratio = t_sec / total_seconds if total_seconds > 0 else 0
        x_pos = int(ratio * w)
        
        cv2.line(frame, (x_pos, y_start), (x_pos, y_start - 5), font_color, 1)
        time_str = f"{t_sec}s"
        cv2.putText(frame, time_str, (x_pos + 2, y_start - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1)

    # 6. 현재 시간 텍스트 및 커서
    cursor_vis_x = min(w - 2, cur_x) # 커서가 화면 밖으로 나가지 않게
    cv2.rectangle(frame, (cursor_vis_x, y_start), (cursor_vis_x + 2, h), (255, 255, 255), -1)
    
    current_seconds = current_frame / fps
    cur_time_str = f"{current_seconds:.1f}s"
    (tw, th), _ = cv2.getTextSize(cur_time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    text_x = cur_x - tw // 2
    text_x = max(0, min(w - tw, text_x))
    text_y = y_start - 25
    
    cv2.rectangle(frame, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
    cv2.putText(frame, cur_time_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# --- [UI 그리기 (하단 배치)] ---
def draw_ui(frame, filename, label_text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    
    # 위험 감지 시 전체 테두리 (타임라인 제외)
    if label_text != "normal" and label_text not in ["initializing", "api error"]:
        cv2.rectangle(frame, (0, 0), (w, h - 30), color, 10)

    # 텍스트 위치: 하단 타임라인 바로 위
    box_bottom = h - 45 
    
    disp_text = label_text.upper()
    
    # 라벨 (Shadow Effect)
    cv2.putText(frame, disp_text, (20, box_bottom), font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, disp_text, (20, box_bottom), font, 1.0, color, 2, cv2.LINE_AA)
    
    # 파일명 정보를 라벨 위에 작게 표시
    info_text = f"{filename[:20]}..."
    cv2.putText(frame, info_text, (20, box_bottom - 35), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, info_text, (20, box_bottom - 35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def get_video_files(directory):
    if not os.path.exists(directory):
        return []
    extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)

# --- [2. AI 추론 쓰레드] ---
def inference_worker(b64_buffer, capture_frame_idx):
    global latest_ui_label, latest_ui_color, is_inferencing, video_segment_predictions, timeline_data
    
    # Updated prompt with 14 categories
    prompt_text = (
        "Classify the video event from the following options:\n"
        "- Abuse\n"
        "- Arrest\n"
        "- Arson\n"
        "- Assault\n"
        "- Burglary\n"
        "- Explosion\n"
        "- Fighting\n"
        "- Normal\n"
        "- RoadAccidents\n"
        "- Robbery\n"
        "- Shooting\n"
        "- Shoplifting\n"
        "- Stealing\n"
        "- Vandalism\n\n"
        "Output ONLY the category name without any extra text or punctuation."
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
            print(f"  [AI] Frame {capture_frame_idx}: {norm.upper()} ({time.time() - start_t:.2f}s)")
            
            latest_ui_label = norm
            latest_ui_color = CLASS_COLORS.get(norm, (100, 100, 100))
            
            video_segment_predictions.append(norm)
            timeline_data.append({'frame': capture_frame_idx, 'label': norm})
        else:
            latest_ui_label = "API ERROR"
            latest_ui_color = CLASS_COLORS["api error"]

    except Exception:
        latest_ui_label = "TIMEOUT"
        latest_ui_color = CLASS_COLORS["api error"]
    
    finally:
        is_inferencing = False

# --- [3. 결과 처리] ---
def decide_final_verdict(predictions):
    if not predictions: return "normal"
    n = len(predictions)
    
    if n < 2:
        for p in predictions:
            if p != "normal": return p
        return "normal"

    detected_candidates = []
    # 1. 연속성
    for i in range(n - 1):
        if predictions[i] != "normal" and predictions[i] == predictions[i+1]:
            detected_candidates.append(predictions[i])
    # 2. 밀도
    if n >= 10:
        for i in range(n - 9):
            window = predictions[i:i+10]
            counts = Counter(window)
            for cls, count in counts.items():
                if cls != "normal" and count >= 3:
                    detected_candidates.append(cls)

    if detected_candidates:
        return Counter(detected_candidates).most_common(1)[0][0]

    # 3. 과반수
    counts = Counter(predictions)
    normal_count = counts["normal"]
    if (n - normal_count) > normal_count:
        del counts["normal"]
        if counts: return counts.most_common(1)[0][0]

    return "normal"

def print_final_report(filenames, y_true, y_pred):
    print("\n" + "="*80)
    print(f"📄 RESULT REPORT ({len(filenames)} Videos)")
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
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    acc = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    labels = [c.title() for c in CLASSES]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.show()

# --- [4. 메인 실행] ---
if __name__ == "__main__":
    video_files = get_video_files(VIDEO_DIR)
    final_data = {'names': [], 'true': [], 'pred': []}

    print(f"▶ Start processing {len(video_files)} videos with 14 classes...")

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        filename = os.path.basename(video_path)
        gt_label = get_ground_truth_from_filename(filename)
        
        # 전체 프레임 가져오기 (메타데이터 신뢰)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1 
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        wait_ms = int(1000 / fps) 

        # --- [초기화] ---
        frame_buffer = deque(maxlen=WINDOW_SIZE)
        video_segment_predictions = [] 
        timeline_data = [] 
        inference_triggers = [] 
        
        latest_ui_label = "SCANNING..."
        latest_ui_color = CLASS_COLORS["initializing"]
        
        last_sample_time = 0 
        sample_count = 0
        current_frame_pos = 0
        last_frame_disp = None 
        
        print(f"\n Playing: {filename} (GT: {gt_label})")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # [보정] 메타데이터보다 실제 프레임이 더 길 경우, total_frames를 늘려줌 (뚫고 나감 방지)
            if current_frame_pos > total_frames:
                total_frames = current_frame_pos

            frame_disp = cv2.resize(frame, (960, 540))
            last_frame_disp = frame_disp.copy() 
            
            if time.time() - last_sample_time > SAMPLE_INTERVAL:
                last_sample_time = time.time()
                frame_buffer.append(frame_to_base64(frame))
                sample_count += 1
                
                if len(frame_buffer) == WINDOW_SIZE and sample_count >= STRIDE and not is_inferencing:
                    is_inferencing = True
                    sample_count = 0
                    
                    inference_triggers.append(current_frame_pos)
                    threading.Thread(target=inference_worker, args=(list(frame_buffer), current_frame_pos), daemon=True).start()

            draw_ui(frame_disp, filename, latest_ui_label, latest_ui_color)
            draw_timeline(frame_disp, current_frame_pos, total_frames, timeline_data, inference_triggers, fps)
            cv2.imshow('Qwen3-VL-2B_multi_class', frame_disp)

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
        
        # ======================================================================
        # [Instant Flush]
        # ======================================================================
        
        pending_inference = is_inferencing
        needs_flush = (len(frame_buffer) > 0)
        
        if pending_inference or needs_flush:
            print(f"  [System] Finalizing analysis...", end="", flush=True)
            
            if not is_inferencing and needs_flush:
                is_inferencing = True
                inference_triggers.append(current_frame_pos) 
                threading.Thread(target=inference_worker, args=(list(frame_buffer), current_frame_pos), daemon=True).start()
            
            if last_frame_disp is not None:
                latest_ui_label = "FINALIZING..." 
                latest_ui_color = (200, 200, 200) 
                
                while is_inferencing:
                    freeze_frame = last_frame_disp.copy()
                    draw_ui(freeze_frame, filename, latest_ui_label, latest_ui_color)
                    
                    # [핵심 수정] 종료 시점에서는 '전체 프레임'을 현재 프레임으로 넣어 꽉 채움
                    draw_timeline(freeze_frame, total_frames, total_frames, timeline_data, inference_triggers, fps)
                    
                    cv2.imshow('Qwen3-VL-2B_multi_class', freeze_frame)
                    cv2.waitKey(50) 

            print(" Done.")
            
            if last_frame_disp is not None:
                final_show = last_frame_disp.copy()
                draw_ui(final_show, filename, latest_ui_label, latest_ui_color)
                # [핵심 수정] 마지막 보여주기 때도 꽉 채운 상태로 그리기
                draw_timeline(final_show, total_frames, total_frames, timeline_data, inference_triggers, fps)
                cv2.imshow('Qwen3-VL-2B_multi_class', final_show)
                cv2.waitKey(1000) # 1초 대기
        
        # ======================================================================

        verdict = decide_final_verdict(video_segment_predictions)
        
        final_data['names'].append(filename)
        final_data['true'].append(gt_label)
        final_data['pred'].append(verdict)
        print(f" Final Decision: {verdict.upper()}")

    cv2.destroyAllWindows()
    print_final_report(final_data['names'], final_data['true'], final_data['pred'])
    plot_confusion_matrix(final_data['true'], final_data['pred'])