from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import time
import threading
from queue import Queue
from collections import defaultdict, deque

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from risk_modules.risk_analyzer import *
from risk_modules.Land_detection import *
from risk_modules.warning_controller import *
import yaml

class LaneTracker:
    def __init__(self, shared_alert_list):
        self.shared_alert = shared_alert_list
        #self.shared_alert = shared_alert
        self.object_history = defaultdict(lambda: deque(maxlen=5))
        self.risk_score_history = defaultdict(lambda: deque(maxlen=10))

        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, 'risk_modules', 'risk_params.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as file:
            self.risk_config = yaml.safe_load(file)['risk_params']

        self.flow_roi_top = self.risk_config['optical_flow']['roi_top_ratio']
        self.flow_roi_bottom = self.risk_config['optical_flow']['roi_bottom_ratio']

        weight_path = os.path.join(current_dir, "weight", "yolov8s.pt")
        self.model = YOLO(weight_path)

        video_path = os.path.join(current_dir, "assets", "road.mp4")
        self.cap = cv2.VideoCapture(video_path)
        self.gray_history = deque(maxlen=3)
        self.prev_smoothed_speed = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.speed_fail_count = 0
        self.MAX_FAIL_COUNT = 3

        self.frame_queue = Queue(maxsize=2)
        self.result_dict = {}  # 用 dict 暫存結果
        self.lock = threading.Lock()
        self.frame_id = 0
        self.next_display_id = 0
        self.stop_event = threading.Event()

        

    def estimate_self_speed(self, prev_gray, curr_gray):
        h, w = curr_gray.shape
        top = int(h * self.flow_roi_top)
        bottom = int(h * self.flow_roi_bottom)
        flow = cv2.calcOpticalFlowFarneback(prev_gray[top:bottom], curr_gray[top:bottom],
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        return np.mean(mag)

    def infer_worker(self):
        while not self.stop_event.is_set():
            if not self.frame_queue.empty():
                fid, frame = self.frame_queue.get()
                try:
                    result = process_frame(frame)
                    frame, _, scene_valid, left_line, right_line = result

                    if not scene_valid:
                        with self.lock:
                            self.result_dict[fid] = frame
                        continue

                    if left_line is not None and len(left_line) == 4:
                        pt1 = (int(left_line[0]), int(left_line[1]))
                        pt2 = (int(left_line[2]), int(left_line[3]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 5)

                    if right_line is not None and len(right_line) == 4:
                        pt1 = (int(right_line[0]), int(right_line[1]))
                        pt2 = (int(right_line[2]), int(right_line[3]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 5)

                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.gray_history.append(curr_gray)

                    if len(self.gray_history) >= 2:
                        speeds = []
                        for i in range(len(self.gray_history) - 1):
                            try:
                                s = self.estimate_self_speed(self.gray_history[i], self.gray_history[i + 1])
                                speeds.append(s)
                            except:
                                pass
                        raw_speed = np.mean(speeds) if speeds else self.prev_smoothed_speed
                        self.speed_fail_count = 0 if speeds else self.speed_fail_count + 1
                    else:
                        raw_speed = self.prev_smoothed_speed
                        self.speed_fail_count += 1

                    alpha = 0.3
                    speed = alpha * raw_speed + (1 - alpha) * self.prev_smoothed_speed

                    if speed < 0.05:
                        speed = self.prev_smoothed_speed
                    if self.speed_fail_count >= self.MAX_FAIL_COUNT:
                        speed *= 0.5

                    self.prev_smoothed_speed = speed
                    roi_dict, scale = get_lane_roi_dynamic(left_line, right_line, frame.shape, speed=speed)

                    results = self.model.track(source=frame, imgsz=1280, persist=True, show=False, stream=False, device=0)
                    risky_objects = []
                    seen_ids = set()
                    annotated_frame = results[0].plot()

                    for r in results[0].boxes.data.cpu().numpy():
                        if len(r) < 7:
                            continue
                        track_id = int(r[6])
                        if track_id in seen_ids:
                            continue
                        seen_ids.add(track_id)

                        x1, y1, x2, y2 = map(int, r[:4])
                        center = get_center((x1, y1, x2, y2))
                        self.object_history[track_id].append(center)
                        obj_speed, is_jump, smoothed_center, vx = compute_speed(track_id, center, self.object_history, fps=self.fps)

                        roi_level = get_roi_level_bbox((x1, y1, x2, y2), roi_dict)
                        if roi_level is None:
                            continue

                        score, level, stay = analyze_risk(track_id, smoothed_center, roi_level, obj_speed, is_jump, vx)
                        self.risk_score_history[track_id].append(score)
                        smoothed_score = np.mean(self.risk_score_history[track_id])

                        if smoothed_score > self.risk_config['score_threshold']['high']:
                            level = "high"
                        elif smoothed_score > self.risk_config['score_threshold']['mid']:
                            level = "mid"
                        else:
                            level = "low"

                        risky_objects.append((x1, y1, x2, y2, track_id, smoothed_score, level))

                        now = time.time()
                        if should_warn(track_id, now, level, smoothed_score, stay, self.risk_config):
                            print(level)
                            if level == "high":
                                self.shared_alert.append("RISK:high")
                            elif level == "mid":
                                self.shared_alert.append("RISK:mid")
                            
                            print(f"⚠️ 提醒觸發！ID={track_id}, Level={level}, Score={smoothed_score:.2f}")
                            cv2.putText(annotated_frame, f"Risk Level: {level.upper()}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    #annotated_frame = results[0].plot()
                    draw_risk_overlay(annotated_frame, risky_objects, roi_dict)
                    cv2.putText(annotated_frame, f"ROI Scale: {scale:.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                    #cv2.putText(annotated_frame, f"Speed: {speed:.3f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    


                    with self.lock:
                        self.result_dict[fid] = annotated_frame

                except Exception as e:
                    print(f"[❌ Inference Exception] {e}")
                    continue

    def run(self):
        self.paused = False  # ➤ 初始為非暫停
        infer_thread = threading.Thread(target=self.infer_worker)
        infer_thread.start()

        while self.cap.isOpened() and not self.stop_event.is_set():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                time.sleep(1 / self.fps)

                if not self.frame_queue.full():
                    self.frame_queue.put((self.frame_id, frame))
                    self.frame_id += 1

                with self.lock:
                    if self.next_display_id in self.result_dict:
                        result_frame = self.result_dict.pop(self.next_display_id)

                        # 顯示處理的偵數
                        cv2.putText(result_frame, f"Frame: {self.next_display_id}", 
                                    (15, 100),  # 位置（你也可以改）
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.9, (255, 255, 255), 2)

                        display_width = 1280
                        display_height = 720
                        resized_frame = cv2.resize(result_frame, (display_width, display_height))
                        cv2.imshow("YOLO + LaneTracker (Ordered)", resized_frame)

                        #cv2.imshow("YOLO + LaneTracker (Ordered)", result_frame)
                        self.next_display_id += 1
            else:
                # 暫停時仍顯示最後一幀（可選）
                #frame_resized = cv2.resize(result_frame, (2560, 1440))
                display_width = 1280
                display_height = 720
                resized_frame = cv2.resize(result_frame, (display_width, display_height))
                cv2.imshow("YOLO + LaneTracker (Ordered)", resized_frame)
                #cv2.imshow("YOLO + LaneTracker (Ordered)", result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break
            elif key == ord(' '):
                self.paused = not self.paused  # ➤ 切換暫停狀態

        self.cap.release()
        cv2.destroyAllWindows()
        infer_thread.join()

