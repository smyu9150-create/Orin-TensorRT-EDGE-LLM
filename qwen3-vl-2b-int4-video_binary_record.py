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
from datetime import datetime

# --- [설정] ---
API_URL = "http://127.0.0.1:8888/v1/chat/completions"
VIDEO_DIR = os.path.expanduser("~/data/video") 
LABELS = ["ABNORMAL", "NORMAL"]

# 로그 저장 디렉토리
LOG_DIR = os.path.expanduser("~/data/inference_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 통합 결과 파일 경로
CONSOLIDATED_LOG = os.path.join(LOG_DIR, f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 2진 분류용 색상 정의 (BGR)
CLASS_COLORS = {
    "NORMAL": (0, 255, 0),       # Green
    "ABNORMAL": (0, 0, 255),     # Red
    "UNCERTAIN": (0, 255, 255),  # Yellow
    "INITIALIZING": (100, 100, 100), # Gray
    "API ERROR": (0, 165, 255),  # Orange
    "TIMEOUT": (128, 128, 128)   # Dark Gray
}

WINDOW_SIZE = 8
STRIDE = 6

# --- [전역 변수] ---
latest_label = "INITIALIZING"
latest_color = CLASS_COLORS["INITIALIZING"]
is_processing = False
last_latency = 0.0
frame_buffer = deque(maxlen=WINDOW_SIZE)
new_frame_count = 0 
last_capture_time = 0

current_video_stats = [] 
timeline_data = []       
inference_triggers = []  
all_results = []

# 인퍼런스 로그 저장용
current_video_inferences = []

def frame_to_base64(frame):
    resized = cv2.resize(frame, (320, 240)) 
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

def draw_timeline(frame, current_frame, total_frames, history, triggers, fps):
    h, w = frame.shape[:2]
    bar_height = 30
    y_start = h - bar_height
    
    if total_frames <= 0 or fps <= 0: return

    progress_ratio = min(1.0, max(0.0, current_frame / total_frames))
    
    cv2.rectangle(frame, (0, y_start), (w, h), (40, 40, 40), -1)

    cur_x = int(progress_ratio * w)

    start_x = 0
    for item in history:
        end_frame = item['frame']
        ratio = min(1.0, end_frame / total_frames)
        end_x = int(ratio * w)
        draw_end_x = min(end_x, cur_x)
        color = CLASS_COLORS.get(item['label'], (100, 100, 100))
        if draw_end_x > start_x:
            cv2.rectangle(frame, (start_x, y_start), (draw_end_x, h), color, -1)
        start_x = end_x
        if start_x >= cur_x: break
            
    if start_x < cur_x:
        waiting_color = CLASS_COLORS["INITIALIZING"]
        cv2.rectangle(frame, (start_x, y_start), (cur_x, h), waiting_color, -1)

    for trig_frame in triggers:
        trig_ratio = min(1.0, trig_frame / total_frames)
        trig_x = int(trig_ratio * w)
        cv2.line(frame, (trig_x, y_start), (trig_x, h), (255, 255, 255), 1)

    total_seconds = total_frames / fps
    step_seconds = max(1, int(total_seconds / 6)) 
    font_color = (200, 200, 200)
    
    for t_sec in range(0, int(total_seconds) + 1, step_seconds):
        ratio = t_sec / total_seconds if total_seconds > 0 else 0
        x_pos = int(ratio * w)
        cv2.line(frame, (x_pos, y_start), (x_pos, y_start - 5), font_color, 1)
        time_str = f"{t_sec}s"
        cv2.putText(frame, time_str, (x_pos + 2, y_start - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1)

    cv2.rectangle(frame, (cur_x, y_start), (min(w, cur_x + 2), h), (255, 255, 255), -1)
    
    current_seconds = current_frame / fps
    cur_time_str = f"{current_seconds:.1f}s"
    (tw, th), _ = cv2.getTextSize(cur_time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    text_x = cur_x - tw // 2
    text_x = max(0, min(w - tw, text_x))
    text_y = y_start - 25
    
    cv2.rectangle(frame, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
    cv2.putText(frame, cur_time_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# --- 인퍼런스 워커 ---
def request_inference(b64_list, capture_frame_idx, fps):
    global latest_label, latest_color, is_processing, last_latency, current_video_stats, timeline_data, current_video_inferences
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
                pred_label = "ABNORMAL"
            elif "NORMAL" in raw_content:
                pred_label = "NORMAL"
            else:
                pred_label = "UNCERTAIN" 

            latest_label = pred_label
            latest_color = CLASS_COLORS.get(pred_label, CLASS_COLORS["UNCERTAIN"])
            
            current_video_stats.append(pred_label)
            timeline_data.append({'frame': capture_frame_idx, 'label': pred_label})

            # 인퍼런스 로그 저장
            timestamp_seconds = capture_frame_idx / fps if fps > 0 else 0
            current_video_inferences.append({
                'frame': capture_frame_idx,
                'timestamp': timestamp_seconds,
                'label': pred_label,
                'latency': latency,
                'raw_response': raw_content
            })

            last_latency = latency
            print(f"  -> [AI Segment] Frame {capture_frame_idx}: {latest_label} ({latency:.2f}s)")
            
        else:
            print(f"Err: {response.status_code}")
            latest_label = "API ERROR"
            latest_color = CLASS_COLORS["API ERROR"]
            
            timestamp_seconds = capture_frame_idx / fps if fps > 0 else 0
            current_video_inferences.append({
                'frame': capture_frame_idx,
                'timestamp': timestamp_seconds,
                'label': "API ERROR",
                'latency': time.time() - start_t,
                'raw_response': f"HTTP {response.status_code}"
            })

    except Exception as e:
        print(f"Req Error: {e}")
        latest_label = "TIMEOUT"
        latest_color = CLASS_COLORS["TIMEOUT"]
        
        timestamp_seconds = capture_frame_idx / fps if fps > 0 else 0
        current_video_inferences.append({
            'frame': capture_frame_idx,
            'timestamp': timestamp_seconds,
            'label': "TIMEOUT",
            'latency': time.time() - start_t,
            'raw_response': str(e)
        })
    
    finally:
        is_processing = False

def save_inference_log(video_filename, inferences, ground_truth, final_verdict, total_frames, fps):
    """
    영상별 인퍼런스 결과를 txt 파일로 저장
    """
    if not inferences:
        print(f"  [Warning] No inferences to save for {video_filename}")
        return
    
    # 파일명 생성 (확장자 제거 + 타임스탬프)
    base_name = os.path.splitext(video_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            # 헤더 정보
            f.write("="*80 + "\n")
            f.write(f"VIDEO INFERENCE LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Video File: {video_filename}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"FPS: {fps:.2f}\n")
            f.write(f"Video Duration: {total_frames/fps:.2f}s\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Final Verdict: {final_verdict}\n")
            f.write(f"Total Inferences: {len(inferences)}\n")
            f.write("="*80 + "\n\n")
            
            # 인퍼런스 결과 상세
            f.write("INFERENCE RESULTS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'#':<4} {'Frame':<8} {'Time(s)':<10} {'Label':<12} {'Latency(s)':<12} {'Raw Response'}\n")
            f.write("-"*80 + "\n")
            
            for idx, inf in enumerate(inferences, 1):
                frame_num = inf['frame']
                timestamp = inf['timestamp']
                label = inf['label']
                latency = inf['latency']
                raw_resp = inf['raw_response'][:50]  # 응답 내용 50자로 제한
                
                f.write(f"{idx:<4} {frame_num:<8} {timestamp:<10.2f} {label:<12} {latency:<12.3f} {raw_resp}\n")
            
            f.write("-"*80 + "\n\n")
            
            # 통계 정보
            normal_count = sum(1 for inf in inferences if inf['label'] == 'NORMAL')
            abnormal_count = sum(1 for inf in inferences if inf['label'] == 'ABNORMAL')
            uncertain_count = sum(1 for inf in inferences if inf['label'] == 'UNCERTAIN')
            error_count = sum(1 for inf in inferences if inf['label'] in ['API ERROR', 'TIMEOUT'])
            
            avg_latency = sum(inf['latency'] for inf in inferences) / len(inferences)
            
            f.write("STATISTICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Normal Predictions: {normal_count} ({normal_count/len(inferences)*100:.1f}%)\n")
            f.write(f"Abnormal Predictions: {abnormal_count} ({abnormal_count/len(inferences)*100:.1f}%)\n")
            f.write(f"Uncertain Predictions: {uncertain_count} ({uncertain_count/len(inferences)*100:.1f}%)\n")
            f.write(f"Errors/Timeouts: {error_count} ({error_count/len(inferences)*100:.1f}%)\n")
            f.write(f"Average Latency: {avg_latency:.3f}s\n")
            f.write("-"*80 + "\n\n")
            
            # 판정 근거
            f.write("VERDICT REASONING:\n")
            f.write("-"*80 + "\n")
            f.write(f"Final Verdict: {final_verdict}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Result: {'✓ CORRECT' if final_verdict == ground_truth else '✗ INCORRECT'}\n")
            f.write("="*80 + "\n")
        
        print(f"  [Log Saved] {log_path}")
        return log_path
        
    except Exception as e:
        print(f"  [Error] Failed to save log for {video_filename}: {e}")
        return None

def save_consolidated_results(video_filename, inferences):
    """
    모든 영상의 윈도우별 예측 결과를 한 파일에 누적 저장
    형식: 영상파일명 LABEL1 LABEL2 LABEL3 ...
    """
    try:
        with open(CONSOLIDATED_LOG, 'a', encoding='utf-8') as f:
            # 윈도우별 레이블만 추출
            labels = [inf['label'] for inf in inferences]
            labels_str = ' '.join(labels)
            
            # 한 줄로 기록: 파일명 + 모든 윈도우 레이블
            f.write(f"{video_filename} {labels_str}\n")
        
        return True
    except Exception as e:
        print(f"  [Error] Failed to append to consolidated log: {e}")
        return False

def get_video_files(directory):
    if not os.path.exists(directory):
        return []
    extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)

# --- UI 그리기 (레이턴시 분리 유지) ---
def draw_ui(frame, filename, label, color, latency):
    h, w = frame.shape[:2]
    
    # 1. 이상행동 감지 시 테두리
    if label == "ABNORMAL":
        cv2.rectangle(frame, (0, 0), (w, h - 30), color, 10) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    box_bottom = h - 45
    
    # 2. 결과 레이블 (좌측 하단)
    cv2.putText(frame, f"{label}", (20, box_bottom), font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"{label}", (20, box_bottom), font, 1.0, color, 2, cv2.LINE_AA)
    
    # 3. 파일명 (레이블 위)
    info = f"{filename[:20]}.."
    cv2.putText(frame, info, (20, box_bottom - 35), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # 4. 레이턴시 표시 (우측 상단)
    lat_text = f"Lat: {latency:.2f}s"
    (lw, lh), _ = cv2.getTextSize(lat_text, font, 0.6, 1)
    lx = w - lw - 20
    ly = 40
    
    cv2.rectangle(frame, (lx - 5, ly - lh - 5), (lx + lw + 5, ly + 5), (0, 0, 0), -1)
    cv2.putText(frame, lat_text, (lx, ly), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

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
    if n >= 10:
        for i in range(n - 9):
            window = stats[i:i+10]
            if window.count("ABNORMAL") >= 3:
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

# --- [메인 실행] ---
if __name__ == "__main__":
    print(f"Searching in: {VIDEO_DIR}")
    print(f"Logs will be saved to: {LOG_DIR}")
    
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
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        wait_ms = int(1000 / fps)

        # 초기화
        frame_buffer.clear()
        current_video_stats = []
        timeline_data = [] 
        inference_triggers = []
        current_video_inferences = []  # 인퍼런스 로그 초기화
        
        latest_label = "SCANNING..."
        latest_color = CLASS_COLORS["INITIALIZING"]
        new_frame_count = 0
        
        last_capture_time = 0 
        
        current_frame_pos = 0
        last_frame_disp = None 

        print(f"\n▶ Playing: {filename} (GT: {true_label})")

        while True:
            ret, frame = cap.read()
            if not ret: break 

            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame_pos > total_frames:
                total_frames = current_frame_pos

            if frame.shape[1] < 640:
                frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            
            last_frame_disp = frame.copy()
            curr_t = time.time()

            # 주기적 샘플링
            if curr_t - last_capture_time > 0.25:
                last_capture_time = curr_t
                img_b64 = frame_to_base64(frame)
                frame_buffer.append(img_b64)
                new_frame_count += 1

                if len(frame_buffer) == WINDOW_SIZE and new_frame_count >= STRIDE:
                    if not is_processing:
                        is_processing = True
                        new_frame_count = 0
                        
                        inference_triggers.append(current_frame_pos)
                        threading.Thread(target=request_inference, args=(list(frame_buffer), current_frame_pos, fps), daemon=True).start()

            draw_ui(frame, filename, latest_label, latest_color, last_latency)
            draw_timeline(frame, current_frame_pos, total_frames, timeline_data, inference_triggers, fps)
            
            cv2.imshow('Qwen3-VL-2B_Binary', frame)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                print_final_summary(all_results)
                plot_beautiful_matrix(all_results)
                exit()
            elif key == ord('n'):
                break 

        cap.release()
        
        # [Instant Flush]
        pending_inference = is_processing
        needs_flush = (len(frame_buffer) > 0)
        
        if pending_inference or needs_flush:
            print(f"  [System] Finalizing analysis...", end="", flush=True)
            
            if not is_processing and needs_flush:
                is_processing = True
                inference_triggers.append(current_frame_pos)
                threading.Thread(target=request_inference, args=(list(frame_buffer), current_frame_pos, fps), daemon=True).start()
            
            if last_frame_disp is not None:
                latest_label = "FINALIZING..."
                latest_color = CLASS_COLORS["INITIALIZING"]
                
                while is_processing:
                    freeze_frame = last_frame_disp.copy()
                    draw_ui(freeze_frame, filename, latest_label, latest_color, last_latency)
                    draw_timeline(freeze_frame, total_frames, total_frames, timeline_data, inference_triggers, fps)
                    cv2.imshow('Qwen3-VL-2B_Binary', freeze_frame)
                    cv2.waitKey(50)
            print(" Done.")
            
            if last_frame_disp is not None:
                final_show = last_frame_disp.copy()
                draw_ui(final_show, filename, latest_label, latest_color, last_latency)
                draw_timeline(final_show, total_frames, total_frames, timeline_data, inference_triggers, fps)
                cv2.imshow('Qwen3-VL-2B_Binary', final_show)
                cv2.waitKey(200) 

        final_verdict = check_final_verdict(current_video_stats)
        
        # 인퍼런스 로그 저장 (개별 상세 로그)
        save_inference_log(filename, current_video_inferences, true_label, final_verdict, total_frames, fps)
        
        # 통합 결과 파일에 추가 (한 줄 요약)
        if current_video_inferences:
            save_consolidated_results(filename, current_video_inferences)
        
        all_results.append({
            'filename': filename,
            'true': true_label,
            'pred': final_verdict
        })
        print(f"✅ Verdict: [{final_verdict}] (GT: {true_label})")

    cv2.destroyAllWindows()
    
    print_final_summary(all_results)
    plot_beautiful_matrix(all_results)
    
    # 통합 결과 파일 위치 안내
    print("\n" + "="*80)
    print(f"📝 All window-level results saved to:")
    print(f"   {CONSOLIDATED_LOG}")
    print("="*80)