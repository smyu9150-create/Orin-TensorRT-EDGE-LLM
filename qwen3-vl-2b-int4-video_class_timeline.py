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
from sklearn.metrics import confusion_matrix
import numpy as np
import re

# --- [설정] ---
API_URL = "http://127.0.0.1:8888/v1/chat/completions"  # 로컬 LLM API 주소 (vLLM 등)
VIDEO_DIR = os.path.expanduser("~/data/video")         # 비디오 경로
# 요청하신 6개 카테고리 정의
LABELS = ["Shooting", "Robbery", "RoadAccident", "Fighting", "Abuse", "Normal"]

# 카테고리별 고유 색상 정의 (BGR)
CLASS_COLORS = {
    "Shooting": (0, 0, 255),       # Red (매우 위험)
    "Robbery": (128, 0, 128),      # Purple
    "RoadAccident": (0, 165, 255), # Orange
    "Fighting": (0, 0, 139),       # Dark Red
    "Abuse": (0, 140, 255),        # Dark Orange
    "Normal": (0, 255, 0),         # Green
    
    # 상태 표시용 색상
    "UNCERTAIN": (0, 255, 255),    # Yellow
    "INITIALIZING": (100, 100, 100), 
    "API ERROR": (0, 0, 0),
    "TIMEOUT": (128, 128, 128)
}

# 요청하신 시스템 프롬프트 상수화
SYSTEM_PROMPT = (
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

def frame_to_base64(frame):
    # 전송 속도를 위해 리사이즈 (필요시 조정)
    resized = cv2.resize(frame, (320, 240)) 
    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

def draw_timeline(frame, current_frame, total_frames, history, triggers, fps):
    h, w = frame.shape[:2]
    bar_height = 30
    y_start = h - bar_height
    
    if total_frames <= 0 or fps <= 0: return

    progress_ratio = min(1.0, max(0.0, current_frame / total_frames))
    
    # 배경
    cv2.rectangle(frame, (0, y_start), (w, h), (40, 40, 40), -1)

    cur_x = int(progress_ratio * w)

    # 과거 기록 그리기
    start_x = 0
    for item in history:
        end_frame = item['frame']
        ratio = min(1.0, end_frame / total_frames)
        end_x = int(ratio * w)
        draw_end_x = min(end_x, cur_x)
        
        # 라벨에 맞는 색상 가져오기
        color = CLASS_COLORS.get(item['label'], CLASS_COLORS["UNCERTAIN"])
        
        if draw_end_x > start_x:
            cv2.rectangle(frame, (start_x, y_start), (draw_end_x, h), color, -1)
        start_x = end_x
        if start_x >= cur_x: break
            
    # 현재 진행 중인 구간 (아직 결과 안나옴)
    if start_x < cur_x:
        waiting_color = CLASS_COLORS["INITIALIZING"]
        cv2.rectangle(frame, (start_x, y_start), (cur_x, h), waiting_color, -1)

    # 트리거 지점 표시
    for trig_frame in triggers:
        trig_ratio = min(1.0, trig_frame / total_frames)
        trig_x = int(trig_ratio * w)
        cv2.line(frame, (trig_x, y_start), (trig_x, h), (255, 255, 255), 1)

    # 시간 텍스트
    total_seconds = total_frames / fps
    step_seconds = max(1, int(total_seconds / 6)) 
    font_color = (200, 200, 200)
    
    for t_sec in range(0, int(total_seconds) + 1, step_seconds):
        ratio = t_sec / total_seconds if total_seconds > 0 else 0
        x_pos = int(ratio * w)
        cv2.line(frame, (x_pos, y_start), (x_pos, y_start - 5), font_color, 1)
        time_str = f"{t_sec}s"
        cv2.putText(frame, time_str, (x_pos + 2, y_start - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1)

    # 현재 위치 커서
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
def request_inference(b64_list, capture_frame_idx):
    global latest_label, latest_color, is_processing, last_latency, current_video_stats, timeline_data
    start_t = time.time()
    
    # 시스템 프롬프트 적용
    content_list = [{"type": "text", "text": SYSTEM_PROMPT}]
    for b64_img in b64_list:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

    payload = {
        "model": "Qwen3-VL-2B-Instruct", # 사용중인 모델명으로 변경 필요 시 수정
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 10, # 단답형이므로 토큰 수 작게
        "temperature": 0.0 
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=8)
        latency = time.time() - start_t
        
        if response.status_code == 200:
            # 출력 정제 (공백, 문장부호 제거)
            raw_content = response.json()['choices'][0]['message']['content'].strip()
            # 특수문자 제거하고 단어만 남김
            clean_content = re.sub(r'[^\w\s]', '', raw_content)
            
            # 사전에 정의된 라벨 중 하나인지 확인 (대소문자 무시 비교)
            pred_label = "UNCERTAIN"
            for valid_label in LABELS:
                if valid_label.lower() == clean_content.lower():
                    pred_label = valid_label
                    break
            
            # 정확히 일치하지 않아도 포함되어 있으면 인정 (예: "It is Shooting" -> "Shooting")
            if pred_label == "UNCERTAIN":
                for valid_label in LABELS:
                    if valid_label.lower() in clean_content.lower():
                        pred_label = valid_label
                        break

            latest_label = pred_label
            latest_color = CLASS_COLORS.get(pred_label, CLASS_COLORS["UNCERTAIN"])
            
            current_video_stats.append(pred_label)
            timeline_data.append({'frame': capture_frame_idx, 'label': pred_label})

            last_latency = latency
            print(f"  -> [AI Segment] Frame {capture_frame_idx}: {latest_label} ({latency:.2f}s)")
            
        else:
            print(f"Err: {response.status_code}")
            latest_label = "API ERROR"
            latest_color = CLASS_COLORS["API ERROR"]

    except Exception as e:
        print(f"Req Error: {e}")
        latest_label = "TIMEOUT"
        latest_color = CLASS_COLORS["TIMEOUT"]
    
    finally:
        is_processing = False

def get_video_files(directory):
    if not os.path.exists(directory):
        return []
    extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)

# --- UI 그리기 ---
def draw_ui(frame, filename, label, color, latency):
    h, w = frame.shape[:2]
    
    # 1. 이상행동(Normal이 아님) 감지 시 테두리
    if label != "Normal" and label in LABELS:
        cv2.rectangle(frame, (0, 0), (w, h - 30), color, 10) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    box_bottom = h - 45
    
    # 2. 결과 레이블 (좌측 하단)
    # 글자 배경 박스
    (lw, lh), _ = cv2.getTextSize(label, font, 1.0, 2)
    cv2.rectangle(frame, (15, box_bottom - lh - 10), (15 + lw + 10, box_bottom + 10), (0,0,0), -1)
    
    cv2.putText(frame, f"{label}", (20, box_bottom), font, 1.0, color, 2, cv2.LINE_AA)
    
    # 3. 파일명 (레이블 위)
    info = f"{filename[:20]}.."
    cv2.putText(frame, info, (20, box_bottom - 40), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # 4. 레이턴시 표시 (우측 상단)
    lat_text = f"Lat: {latency:.2f}s"
    (lw, lh), _ = cv2.getTextSize(lat_text, font, 0.6, 1)
    lx = w - lw - 20
    ly = 40
    
    cv2.rectangle(frame, (lx - 5, ly - lh - 5), (lx + lw + 5, ly + 5), (0, 0, 0), -1)
    cv2.putText(frame, lat_text, (lx, ly), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

# 파일명을 보고 정답(Ground Truth)을 유추하는 함수
def get_ground_truth(filename):
    fname_lower = filename.lower()
    
    # 정의된 라벨이 파일명에 포함되어 있는지 확인
    for label in LABELS:
        if label.lower() in fname_lower:
            return label
            
    # 별칭 처리 (파일명이 정확하지 않을 경우)
    if "gun" in fname_lower or "fire" in fname_lower: return "Shooting"
    if "steal" in fname_lower or "thief" in fname_lower: return "Robbery"
    if "crash" in fname_lower or "car" in fname_lower: return "RoadAccident"
    if "punch" in fname_lower or "brawl" in fname_lower: return "Fighting"
    if "hit" in fname_lower or "slap" in fname_lower: return "Abuse"
    
    return "Normal" # 기본값

# 다중 클래스 최종 판단 로직 (최빈값 또는 위험 우선순위)
def check_final_verdict(stats):
    if not stats: return "Normal"
    
    # 카운트
    counts = Counter(stats)
    
    # 1. Normal이 압도적인지 확인 (80% 이상)
    total = len(stats)
    if counts["Normal"] / total > 0.8:
        return "Normal"
        
    # 2. Normal 제외하고 가장 많이 나온 라벨 찾기
    non_normal_stats = [s for s in stats if s != "Normal" and s in LABELS]
    
    if not non_normal_stats:
        return "Normal"
        
    # 가장 많이 등장한 위험 행동 반환
    most_common = Counter(non_normal_stats).most_common(1)
    return most_common[0][0]

def plot_beautiful_matrix(results):
    if not results: return
    y_true = [r['true'] for r in results]
    y_pred = [r['pred'] for r in results]
    
    # 모든 라벨이 포함되도록 설정
    unique_labels = sorted(list(set(y_true + y_pred + LABELS)))
    # LABELS 순서대로 정렬 (Normal을 마지막으로)
    ordered_labels = [l for l in LABELS if l in unique_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                     xticklabels=ordered_labels, yticklabels=ordered_labels,
                     annot_kws={"size": 14}, cbar=False)
    plt.title(f"Confusion Matrix (Acc: {accuracy:.1%})", fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_final_summary(results):
    if not results: return
    print("\n" + "="*90)
    print(f"📄 FINAL PROCESSING REPORT ({len(results)} Videos)")
    print("="*90)
    print(f"{'FILENAME':<30} | {'PREDICT':<15} | {'TRUTH':<15} | {'RESULT'}")
    print("-" * 90)
    correct_cnt = 0
    for res in results:
        fname = res['filename']
        if len(fname) > 28: fname = fname[:25] + "..."
        pred = res['pred']
        gt = res['true']
        is_correct = (pred == gt)
        if is_correct: 
            correct_cnt += 1
            mark = "✅ OK"
        else:
            mark = "❌ FAIL"
        print(f"{fname:<30} | {pred:<15} | {gt:<15} | {mark}")
    acc = correct_cnt / len(results) * 100
    print("-" * 90)
    print(f"📊 Total Accuracy: {acc:.2f}% ({correct_cnt}/{len(results)})")
    print("="*90 + "\n")

# --- [메인 실행] ---
if __name__ == "__main__":
    print(f"Searching in: {VIDEO_DIR}")
    video_files = get_video_files(VIDEO_DIR)
    
    if not video_files:
        print(f"❌ Error: No videos found in {VIDEO_DIR}")
        print("Please check the VIDEO_DIR path.")
        exit()

    print(f"📂 Found {len(video_files)} videos.")
    print(f"🎯 Target Classes: {LABELS}")
    
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

            # 디스플레이용 리사이즈
            if frame.shape[1] < 640:
                frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            
            last_frame_disp = frame.copy()
            curr_t = time.time()

            # 샘플링 및 인퍼런스 요청
            if curr_t - last_capture_time > 0.25: # 초당 4회 샘플링 시도
                last_capture_time = curr_t
                img_b64 = frame_to_base64(frame)
                frame_buffer.append(img_b64)
                new_frame_count += 1

                if len(frame_buffer) == WINDOW_SIZE and new_frame_count >= STRIDE:
                    if not is_processing:
                        is_processing = True
                        new_frame_count = 0
                        
                        inference_triggers.append(current_frame_pos)
                        threading.Thread(target=request_inference, args=(list(frame_buffer), current_frame_pos), daemon=True).start()

            draw_ui(frame, filename, latest_label, latest_color, last_latency)
            draw_timeline(frame, current_frame_pos, total_frames, timeline_data, inference_triggers, fps)
            
            cv2.imshow('Multi-Class Safety AI', frame)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                print_final_summary(all_results)
                plot_beautiful_matrix(all_results)
                exit()
            elif key == ord('n'): # 다음 비디오로 넘기기
                break 

        cap.release()
        
        # [마지막 버퍼 처리]
        pending_inference = is_processing
        needs_flush = (len(frame_buffer) > 0)
        
        if pending_inference or needs_flush:
            print(f"  [System] Finalizing analysis...", end="", flush=True)
            
            if not is_processing and needs_flush:
                is_processing = True
                inference_triggers.append(current_frame_pos)
                threading.Thread(target=request_inference, args=(list(frame_buffer), current_frame_pos), daemon=True).start()
            
            if last_frame_disp is not None:
                latest_label = "FINALIZING..."
                latest_color = CLASS_COLORS["INITIALIZING"]
                
                # 처리 완료될 때까지 대기하며 화면 갱신
                while is_processing:
                    freeze_frame = last_frame_disp.copy()
                    draw_ui(freeze_frame, filename, latest_label, latest_color, last_latency)
                    draw_timeline(freeze_frame, total_frames, total_frames, timeline_data, inference_triggers, fps)
                    cv2.imshow('Multi-Class Safety AI', freeze_frame)
                    cv2.waitKey(50)
            print(" Done.")
            
            # 최종 프레임 잠깐 보여주기
            if last_frame_disp is not None:
                final_show = last_frame_disp.copy()
                draw_ui(final_show, filename, latest_label, latest_color, last_latency)
                draw_timeline(final_show, total_frames, total_frames, timeline_data, inference_triggers, fps)
                cv2.imshow('Multi-Class Safety AI', final_show)
                cv2.waitKey(500) 

        final_verdict = check_final_verdict(current_video_stats)
        
        all_results.append({
            'filename': filename,
            'true': true_label,
            'pred': final_verdict
        })
        print(f"✅ Verdict: [{final_verdict}] (GT: {true_label})")

    cv2.destroyAllWindows()
    
    print_final_summary(all_results)
    plot_beautiful_matrix(all_results)