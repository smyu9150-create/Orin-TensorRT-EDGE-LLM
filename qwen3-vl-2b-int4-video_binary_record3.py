import cv2
import requests
import base64
import threading
import time
import os
import glob
from collections import deque
import matplotlib
matplotlib.use('Agg')  # 백그라운드 모드로 변경 (GUI 없이 파일로만 저장)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime
import psutil
import gc
import signal
import sys
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

# --- [설정] ---
API_URL = "http://127.0.0.1:8888/v1/chat/completions"
VIDEO_DIR = os.path.expanduser("~/data/video") 
LABELS = ["ABNORMAL", "NORMAL"]

# 장기 실행 최적화 설정
HEADLESS_MODE = False  # True로 설정하면 GUI 없이 실행 (서버 환경용)
MAX_CONCURRENT_INFERENCES = 2  # 동시 인퍼런스 제한
RESULTS_BACKUP_INTERVAL = 100  # 100개마다 results 백업
RESULTS_KEEP_IN_MEMORY = 50  # 메모리에 최근 50개만 유지

# 로그 저장 디렉토리
LOG_DIR = os.path.expanduser("~/data/inference_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 체크포인트 디렉토리 (중단된 위치 저장)
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 백업 디렉토리
BACKUP_DIR = os.path.join(LOG_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

# 타임스탬프 생성 (실행 시작 시 한 번만)
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 통합 결과 파일 경로
CONSOLIDATED_LOG = os.path.join(LOG_DIR, f"all_results_{RUN_TIMESTAMP}.txt")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, f"progress_{RUN_TIMESTAMP}.json")
ERROR_LOG = os.path.join(LOG_DIR, f"errors_{RUN_TIMESTAMP}.txt")

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

# API 재시도 설정
MAX_API_RETRIES = 3
API_RETRY_DELAY = 2  # seconds
API_TIMEOUT = 15  # seconds (장기 실행용으로 증가)

# 메모리 관리 설정
MEMORY_CHECK_INTERVAL = 10  # 10개 비디오마다 메모리 체크
MEMORY_THRESHOLD_PERCENT = 85  # 85% 이상 사용 시 경고
GC_COLLECT_INTERVAL = 5  # 5개 비디오마다 강제 가비지 컬렉션
DISK_SPACE_CHECK_INTERVAL = 50  # 50개마다 디스크 공간 체크
DISK_SPACE_THRESHOLD = 90  # 90% 이상 사용 시 경고

# GUI 리소스 관리
GUI_REFRESH_INTERVAL = 50  # 50개마다 GUI 윈도우 재생성

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

# 프로그램 종료 플래그
shutdown_flag = False

# 통계 정보
stats = {
    'total_processed': 0,
    'total_errors': 0,
    'api_timeouts': 0,
    'api_errors': 0,
    'videos_skipped': 0,
    'start_time': time.time(),
    'last_backup_time': time.time(),
    'total_backups': 0
}

# 스레드 풀 executor
inference_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_INFERENCES)

def signal_handler(sig, frame):
    """Ctrl+C 등 종료 시그널 처리"""
    global shutdown_flag
    print("\n\n⚠️  Shutdown signal received. Saving progress...")
    shutdown_flag = True
    save_checkpoint()
    backup_results(force=True)
    inference_executor.shutdown(wait=True, cancel_futures=True)
    print("✅ Progress saved. Exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log_error(error_msg, video_filename=""):
    """에러 로그 기록"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(ERROR_LOG, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {video_filename}: {error_msg}\n")
        print(f"❌ Error logged: {error_msg}")
    except Exception as e:
        print(f"⚠️ Failed to write error log: {e}")

def check_memory_usage():
    """메모리 사용량 확인 및 경고"""
    try:
        memory = psutil.virtual_memory()
        percent = memory.percent
        
        if percent > MEMORY_THRESHOLD_PERCENT:
            print(f"⚠️  High memory usage: {percent:.1f}% - Running garbage collection...")
            gc.collect()
            memory_after = psutil.virtual_memory()
            print(f"   Memory after GC: {memory_after.percent:.1f}%")
        
        return percent
    except Exception as e:
        print(f"⚠️ Memory check failed: {e}")
        return 0

def check_disk_space():
    """디스크 공간 확인"""
    try:
        disk = psutil.disk_usage(LOG_DIR)
        percent = disk.percent
        
        if percent > DISK_SPACE_THRESHOLD:
            print(f"⚠️  WARNING: Disk space critical - {percent:.1f}% used!")
            print(f"   Free space: {disk.free / (1024**3):.2f} GB")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️ Disk space check failed: {e}")
        return True

def backup_results(force=False):
    """메모리에서 results를 디스크로 백업하고 정리"""
    global all_results
    
    if not all_results:
        return
    
    try:
        backup_file = os.path.join(BACKUP_DIR, f"results_backup_{stats['total_processed']}_{RUN_TIMESTAMP}.json")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        stats['total_backups'] += 1
        stats['last_backup_time'] = time.time()
        print(f"💾 Backed up {len(all_results)} results to: {backup_file}")
        
        if not force:
            # 최근 결과만 메모리에 유지
            all_results = all_results[-RESULTS_KEEP_IN_MEMORY:]
            print(f"   Kept {len(all_results)} most recent results in memory")
        
        gc.collect()
        
    except Exception as e:
        log_error(f"Failed to backup results: {e}")

def load_all_backups():
    """모든 백업 파일에서 results 로드"""
    all_backup_results = []
    
    try:
        backup_files = sorted(glob.glob(os.path.join(BACKUP_DIR, f"results_backup_*_{RUN_TIMESTAMP}.json")))
        
        for backup_file in backup_files:
            with open(backup_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_backup_results.extend(results)
        
        print(f"📂 Loaded {len(all_backup_results)} results from {len(backup_files)} backup files")
        return all_backup_results
        
    except Exception as e:
        print(f"⚠️ Failed to load backups: {e}")
        return []

def save_checkpoint():
    """현재 진행 상황 저장"""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'processed_count': stats['total_processed'],
        'completed_videos': [r['filename'] for r in all_results],
        'stats': stats,
        'run_id': RUN_TIMESTAMP
    }
    
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved: {stats['total_processed']} videos processed")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")
        log_error(f"Checkpoint save failed: {e}")

def load_checkpoint():
    """이전 체크포인트 로드"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"📂 Found checkpoint: {len(checkpoint['completed_videos'])} videos already processed")
        return checkpoint
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None

def frame_to_base64(frame):
    """프레임을 base64로 변환 (에러 처리 추가)"""
    try:
        resized = cv2.resize(frame, (320, 240)) 
        _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        log_error(f"Frame encoding error: {e}")
        return None

def draw_timeline(frame, current_frame, total_frames, history, triggers, fps):
    """타임라인 그리기 (에러 방지)"""
    try:
        h, w = frame.shape[:2]
        bar_height = 30
        y_start = h - bar_height
        
        if total_frames <= 0 or fps <= 0: 
            return

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
            if start_x >= cur_x: 
                break
                
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
    except Exception as e:
        # 타임라인 그리기 실패해도 계속 진행
        pass

# --- 인퍼런스 워커 (재시도 로직 추가) ---
def request_inference(b64_list, capture_frame_idx, fps, retry_count=0):
    """API 요청 with 재시도 로직"""
    global latest_label, latest_color, is_processing, last_latency, current_video_stats, timeline_data, current_video_inferences, stats
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
        if b64_img:  # None 체크
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

    payload = {
        "model": "Qwen3-VL-2B-Instruct", 
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 5, 
        "temperature": 0.0 
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=API_TIMEOUT)
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
            # HTTP 에러 시 재시도
            if retry_count < MAX_API_RETRIES:
                print(f"⚠️  API Error {response.status_code}, retrying ({retry_count + 1}/{MAX_API_RETRIES})...")
                time.sleep(API_RETRY_DELAY)
                return request_inference(b64_list, capture_frame_idx, fps, retry_count + 1)
            
            print(f"❌ API Error after {MAX_API_RETRIES} retries: {response.status_code}")
            stats['api_errors'] += 1
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

    except requests.exceptions.Timeout:
        # 타임아웃 시 재시도
        if retry_count < MAX_API_RETRIES:
            print(f"⚠️  API Timeout, retrying ({retry_count + 1}/{MAX_API_RETRIES})...")
            time.sleep(API_RETRY_DELAY)
            return request_inference(b64_list, capture_frame_idx, fps, retry_count + 1)
        
        print(f"❌ API Timeout after {MAX_API_RETRIES} retries")
        stats['api_timeouts'] += 1
        latest_label = "TIMEOUT"
        latest_color = CLASS_COLORS["TIMEOUT"]
        
        timestamp_seconds = capture_frame_idx / fps if fps > 0 else 0
        current_video_inferences.append({
            'frame': capture_frame_idx,
            'timestamp': timestamp_seconds,
            'label': "TIMEOUT",
            'latency': time.time() - start_t,
            'raw_response': "Request timeout"
        })
    
    except Exception as e:
        # 기타 예외 시 재시도
        if retry_count < MAX_API_RETRIES:
            print(f"⚠️  Request Error: {e}, retrying ({retry_count + 1}/{MAX_API_RETRIES})...")
            time.sleep(API_RETRY_DELAY)
            return request_inference(b64_list, capture_frame_idx, fps, retry_count + 1)
        
        print(f"❌ Request Error after {MAX_API_RETRIES} retries: {e}")
        stats['total_errors'] += 1
        log_error(f"Inference request failed: {e}")
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
    """영상별 인퍼런스 결과를 txt 파일로 저장"""
    if not inferences:
        print(f"  [Warning] No inferences to save for {video_filename}")
        return
    
    base_name = os.path.splitext(video_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
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
            
            f.write("INFERENCE RESULTS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'#':<4} {'Frame':<8} {'Time(s)':<10} {'Label':<12} {'Latency(s)':<12} {'Raw Response'}\n")
            f.write("-"*80 + "\n")
            
            for idx, inf in enumerate(inferences, 1):
                frame_num = inf['frame']
                timestamp = inf['timestamp']
                label = inf['label']
                latency = inf['latency']
                raw_resp = inf['raw_response'][:50]
                
                f.write(f"{idx:<4} {frame_num:<8} {timestamp:<10.2f} {label:<12} {latency:<12.3f} {raw_resp}\n")
            
            f.write("-"*80 + "\n\n")
            
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
            
            f.write("VERDICT REASONING:\n")
            f.write("-"*80 + "\n")
            f.write(f"Final Verdict: {final_verdict}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Result: {'✓ CORRECT' if final_verdict == ground_truth else '✗ INCORRECT'}\n")
            f.write("="*80 + "\n")
        
        return log_path
        
    except Exception as e:
        log_error(f"Failed to save log for {video_filename}: {e}")
        return None

def save_consolidated_results(video_filename, inferences):
    """모든 영상의 윈도우별 예측 결과를 한 파일에 누적 저장"""
    try:
        labels = [inf['label'] for inf in inferences]
        labels_str = ' '.join(labels)
        
        with open(CONSOLIDATED_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{video_filename} {labels_str}\n")
        
        return True
    except Exception as e:
        log_error(f"Failed to append to consolidated log: {e}")
        return False

def get_video_files(directory):
    """비디오 파일 목록 가져오기"""
    if not os.path.exists(directory):
        return []
    extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)

def draw_ui(frame, filename, label, color, latency):
    """UI 그리기 (에러 방지)"""
    try:
        h, w = frame.shape[:2]
        
        if label == "ABNORMAL":
            cv2.rectangle(frame, (0, 0), (w, h - 30), color, 10) 

        font = cv2.FONT_HERSHEY_SIMPLEX
        box_bottom = h - 45
        
        cv2.putText(frame, f"{label}", (20, box_bottom), font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{label}", (20, box_bottom), font, 1.0, color, 2, cv2.LINE_AA)
        
        info = f"{filename[:20]}.."
        cv2.putText(frame, info, (20, box_bottom - 35), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        lat_text = f"Lat: {latency:.2f}s"
        (lw, lh), _ = cv2.getTextSize(lat_text, font, 0.6, 1)
        lx = w - lw - 20
        ly = 40
        
        cv2.rectangle(frame, (lx - 5, ly - lh - 5), (lx + lw + 5, ly + 5), (0, 0, 0), -1)
        cv2.putText(frame, lat_text, (lx, ly), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception as e:
        pass  # UI 그리기 실패해도 계속 진행

def get_ground_truth(filename):
    """파일명에서 Ground Truth 추출"""
    fname_lower = filename.lower()
    if "normal" in fname_lower:
        return "NORMAL"
    else:
        return "ABNORMAL"

def check_final_verdict(stats):
    """최종 판정 로직"""
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
    """Confusion Matrix 플롯 (파일로 저장)"""
    if not results: 
        return
    
    try:
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
        
        # 파일로 저장
        plot_path = os.path.join(LOG_DIR, f"confusion_matrix_{RUN_TIMESTAMP}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved: {plot_path}")
        
    except Exception as e:
        log_error(f"Failed to create confusion matrix: {e}")

def print_final_summary(results):
    """최종 요약 출력"""
    if not results: 
        return
    
    print("\n" + "="*80)
    print(f"📄 FINAL PROCESSING REPORT ({len(results)} Videos)")
    print("="*80)
    print(f"{'FILENAME':<35} | {'PREDICT':<10} | {'TRUTH':<10} | {'RESULT'}")
    print("-" * 80)
    
    correct_cnt = 0
    for res in results:
        fname = res['filename']
        if len(fname) > 33: 
            fname = fname[:30] + "..."
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
    elapsed_time = time.time() - stats['start_time']
    elapsed_hours = elapsed_time / 3600
    
    print("-" * 80)
    print(f"📊 Total Accuracy: {acc:.2f}% ({correct_cnt}/{len(results)})")
    print(f"⏱️  Elapsed Time: {elapsed_hours:.2f} hours ({elapsed_time/86400:.2f} days)")
    print(f"🎬 Videos Processed: {stats['total_processed']}")
    print(f"⚠️  API Timeouts: {stats['api_timeouts']}")
    print(f"❌ API Errors: {stats['api_errors']}")
    print(f"⏭️  Videos Skipped: {stats['videos_skipped']}")
    print(f"💾 Total Backups: {stats['total_backups']}")
    print("="*80 + "\n")

def print_progress_stats():
    """진행 상황 통계 출력"""
    elapsed = time.time() - stats['start_time']
    elapsed_hours = elapsed / 3600
    elapsed_days = elapsed / 86400
    memory_percent = psutil.virtual_memory().percent
    
    try:
        disk = psutil.disk_usage(LOG_DIR)
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024**3)
    except:
        disk_percent = 0
        disk_free_gb = 0
    
    print("\n" + "-"*80)
    print(f"⏱️  Runtime: {elapsed_hours:.2f}h ({elapsed_days:.2f} days) | Processed: {stats['total_processed']} videos")
    print(f"📊 Memory: {memory_percent:.1f}% | Disk: {disk_percent:.1f}% ({disk_free_gb:.1f}GB free)")
    print(f"❌ Errors: {stats['total_errors']} | Timeouts: {stats['api_timeouts']} | Backups: {stats['total_backups']}")
    print("-"*80 + "\n")

def refresh_gui_window():
    """GUI 윈도우 리소스 정리 및 재생성"""
    if not HEADLESS_MODE:
        try:
            cv2.destroyAllWindows()
            time.sleep(0.5)
            print("🔄 GUI window refreshed")
        except Exception as e:
            print(f"⚠️ GUI refresh failed: {e}")

# --- [메인 실행] ---
if __name__ == "__main__":
    print("="*80)
    print("🚀 ULTRA LONG-RUN VIDEO ANALYSIS SYSTEM (128h Optimized)")
    print("="*80)
    print(f"📂 Video Directory: {VIDEO_DIR}")
    print(f"💾 Log Directory: {LOG_DIR}")
    print(f"🔄 Checkpoint File: {CHECKPOINT_FILE}")
    print(f"⚙️  API Timeout: {API_TIMEOUT}s | Max Retries: {MAX_API_RETRIES}")
    print(f"🧵 Max Concurrent Inferences: {MAX_CONCURRENT_INFERENCES}")
    print(f"💾 Results Backup: Every {RESULTS_BACKUP_INTERVAL} videos")
    print(f"🖥️  Headless Mode: {'ENABLED' if HEADLESS_MODE else 'DISABLED'}")
    print("="*80 + "\n")
    
    # 체크포인트 로드
    checkpoint = load_checkpoint()
    completed_videos = set(checkpoint['completed_videos']) if checkpoint else set()
    
    video_files = get_video_files(VIDEO_DIR)
    
    if not video_files:
        print(f"❌ Error: No videos found in {VIDEO_DIR}")
        exit()

    print(f"📂 Found {len(video_files)} total videos.")
    
    if completed_videos:
        remaining_videos = [v for v in video_files if os.path.basename(v) not in completed_videos]
        print(f"✅ Already completed: {len(completed_videos)} videos")
        print(f"📋 Remaining: {len(remaining_videos)} videos")
        video_files = remaining_videos
    
    if not video_files:
        print("✅ All videos already processed!")
        exit()
    
    print(f"\n▶️  Starting processing of {len(video_files)} videos...\n")
    
    for video_idx, video_path in enumerate(video_files, 1):
        if shutdown_flag:
            print("\n⚠️  Shutdown requested. Stopping...")
            break
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  Failed to open video: {video_path}")
                stats['videos_skipped'] += 1
                log_error(f"Failed to open video", os.path.basename(video_path))
                continue

            filename = os.path.basename(video_path)
            true_label = get_ground_truth(filename)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: 
                total_frames = 1
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: 
                fps = 30
            wait_ms = max(1, int(1000 / fps))

            # 초기화
            frame_buffer.clear()
            current_video_stats = []
            timeline_data = [] 
            inference_triggers = []
            current_video_inferences = []
            
            latest_label = "SCANNING..."
            latest_color = CLASS_COLORS["INITIALIZING"]
            new_frame_count = 0
            last_capture_time = 0 
            current_frame_pos = 0
            last_frame_disp = None 

            print(f"\n[{video_idx}/{len(video_files)}] ▶ Playing: {filename} (GT: {true_label})")

            # 비디오 처리 루프
            while True:
                if shutdown_flag:
                    break
                
                ret, frame = cap.read()
                if not ret: 
                    break 

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
                    if img_b64:  # None이 아닐 때만 추가
                        frame_buffer.append(img_b64)
                        new_frame_count += 1

                    if len(frame_buffer) == WINDOW_SIZE and new_frame_count >= STRIDE:
                        if not is_processing:
                            is_processing = True
                            new_frame_count = 0
                            
                            inference_triggers.append(current_frame_pos)
                            
                            # ThreadPoolExecutor 사용
                            inference_executor.submit(request_inference, list(frame_buffer), current_frame_pos, fps)

                if not HEADLESS_MODE:
                    draw_ui(frame, filename, latest_label, latest_color, last_latency)
                    draw_timeline(frame, current_frame_pos, total_frames, timeline_data, inference_triggers, fps)
                    
                    cv2.imshow('Safe Long-Run Video Analysis', frame)

                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord('q'): 
                        shutdown_flag = True
                        break
                    elif key == ord('n'):
                        break 
                else:
                    # Headless mode: 주기적으로 진행 상황만 출력
                    if current_frame_pos % 300 == 0:  # 10초마다
                        progress = (current_frame_pos / total_frames) * 100
                        print(f"  Progress: {progress:.1f}% ({current_frame_pos}/{total_frames} frames)")

            cap.release()
            
            # 마지막 인퍼런스 처리
            if not shutdown_flag:
                pending_inference = is_processing
                needs_flush = (len(frame_buffer) > 0)
                
                if pending_inference or needs_flush:
                    print(f"  [System] Finalizing analysis...", end="", flush=True)
                    
                    if not is_processing and needs_flush:
                        is_processing = True
                        inference_triggers.append(current_frame_pos)
                        inference_executor.submit(request_inference, list(frame_buffer), current_frame_pos, fps)
                    
                    # 최대 30초 대기
                    wait_start = time.time()
                    while is_processing and (time.time() - wait_start < 30):
                        if not HEADLESS_MODE and last_frame_disp is not None:
                            freeze_frame = last_frame_disp.copy()
                            draw_ui(freeze_frame, filename, "FINALIZING...", CLASS_COLORS["INITIALIZING"], last_latency)
                            draw_timeline(freeze_frame, total_frames, total_frames, timeline_data, inference_triggers, fps)
                            cv2.imshow('Safe Long-Run Video Analysis', freeze_frame)
                            cv2.waitKey(50)
                        else:
                            time.sleep(0.5)
                    
                    print(" Done.")

                # 결과 저장
                final_verdict = check_final_verdict(current_video_stats)
                
                save_inference_log(filename, current_video_inferences, true_label, final_verdict, total_frames, fps)
                
                if current_video_inferences:
                    save_consolidated_results(filename, current_video_inferences)
                
                all_results.append({
                    'filename': filename,
                    'true': true_label,
                    'pred': final_verdict
                })
                
                stats['total_processed'] += 1
                print(f"✅ Verdict: [{final_verdict}] (GT: {true_label})")
            
            # 주기적인 메모리 체크 및 가비지 컬렉션
            if video_idx % MEMORY_CHECK_INTERVAL == 0:
                check_memory_usage()
            
            if video_idx % GC_COLLECT_INTERVAL == 0:
                gc.collect()
            
            # 디스크 공간 체크
            if video_idx % DISK_SPACE_CHECK_INTERVAL == 0:
                if not check_disk_space():
                    print("⚠️  WARNING: Consider cleaning up old logs or expanding disk space!")
            
            # Results 백업
            if video_idx % RESULTS_BACKUP_INTERVAL == 0:
                backup_results()
            
            # 주기적인 체크포인트 저장 (10개마다)
            if video_idx % 10 == 0:
                save_checkpoint()
                print_progress_stats()
            
            # GUI 윈도우 리프레시
            if not HEADLESS_MODE and video_idx % GUI_REFRESH_INTERVAL == 0:
                refresh_gui_window()
        
        except Exception as e:
            print(f"\n❌ Critical error processing {os.path.basename(video_path)}: {e}")
            log_error(f"Critical error: {traceback.format_exc()}", os.path.basename(video_path))
            stats['total_errors'] += 1
            stats['videos_skipped'] += 1
            
            # 에러 발생해도 다음 비디오로 계속 진행
            try:
                cap.release()
            except:
                pass
            continue

    # 정리
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    
    # 스레드 풀 종료
    inference_executor.shutdown(wait=True)
    
    # 최종 백업
    backup_results(force=True)
    
    # 모든 백업 로드하여 최종 리포트 생성
    all_backup_results = load_all_backups()
    if all_backup_results:
        all_results = all_backup_results
    
    # 최종 결과 저장 및 출력
    save_checkpoint()
    print_final_summary(all_results)
    plot_beautiful_matrix(all_results)
    
    print("\n" + "="*80)
    print(f"📝 All window-level results saved to:")
    print(f"   {CONSOLIDATED_LOG}")
    print(f"💾 Checkpoint saved to:")
    print(f"   {CHECKPOINT_FILE}")
    print(f"💾 Results backups in:")
    print(f"   {BACKUP_DIR}")
    print(f"❌ Error log saved to:")
    print(f"   {ERROR_LOG}")
    print("="*80)
    
    elapsed = time.time() - stats['start_time']
    print(f"\n✅ Long-run processing completed successfully!")
    print(f"⏱️  Total Runtime: {elapsed/3600:.2f} hours ({elapsed/86400:.2f} days)")