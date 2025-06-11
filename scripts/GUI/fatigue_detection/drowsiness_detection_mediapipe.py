import cv2
import mediapipe as mp
import numpy as np
import time

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(eye_landmarks):
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_landmarks):
    A = euclidean_distance(mouth_landmarks[13], mouth_landmarks[19])
    B = euclidean_distance(mouth_landmarks[14], mouth_landmarks[18])
    C = euclidean_distance(mouth_landmarks[12], mouth_landmarks[16])
    return (A + B) / (2.0 * C)

def start_drowsiness_detection(shared_alert_list):
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    TIRED_SECONDS = 2.0
    EAR_CALIBRATION_TIME = 3
    MAR_OPEN_THRESHOLD = 1.4
    MAR_CLOSE_THRESHOLD = 0.7
    YAWN_ALERT_THRESHOLD = 3
    ALERT_DISPLAY_DURATION = 3  # 警示停留秒數

    # 初始化校準與狀態
    calibration_ears = []
    calibration_done = False
    calibration_start = time.time()
    EAR_THRESHOLD = None
    YAWN_COUNT = 0
    yawn_flag = False
    eye_close_start_time = None
    last_alert_time = 0
    last_alert_message = ""

    def reset_state():
        nonlocal calibration_ears, calibration_done, calibration_start, EAR_THRESHOLD
        nonlocal YAWN_COUNT, yawn_flag, eye_close_start_time
        nonlocal last_alert_time, last_alert_message
        calibration_ears = []
        calibration_done = False
        calibration_start = time.time()
        EAR_THRESHOLD = None
        YAWN_COUNT = 0
        yawn_flag = False
        eye_close_start_time = None
        last_alert_time = 0
        last_alert_message = ""
        print("[INFO] 校準與狀態已重置")

    reset_state()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 無法從攝影機取得影像，退出")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        current_time = time.time()

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]

            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            mouth = [landmarks[i] for i in [78, 81, 13, 311, 308, 402, 14, 87, 95, 88, 178, 317, 82, 81, 80, 191, 88, 178, 87, 14]]

            mouth_mar_landmarks = {
                12: mouth[5], 13: mouth[2], 14: mouth[6],
                16: mouth[4], 18: mouth[17], 19: mouth[19]
            }

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth_mar_landmarks)

            if not calibration_done:
                calibration_ears.append(ear)
                elapsed = current_time - calibration_start
                remaining = max(0, int(EAR_CALIBRATION_TIME - elapsed))
                cv2.putText(frame, f"Calibrating... {remaining}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if elapsed >= EAR_CALIBRATION_TIME:
                    EAR_THRESHOLD = np.mean(calibration_ears) * 0.75
                    calibration_done = True
            else:
                if ear < EAR_THRESHOLD:
                    if eye_close_start_time is None:
                        eye_close_start_time = current_time
                    elif current_time - eye_close_start_time >= TIRED_SECONDS:
                        shared_alert_list.append("DROWSINESS")
                        last_alert_time = current_time
                        last_alert_message = "DROWSINESS"
                else:
                    eye_close_start_time = None

                if mar > MAR_OPEN_THRESHOLD and not yawn_flag:
                    YAWN_COUNT += 1
                    yawn_flag = True
                elif mar < MAR_CLOSE_THRESHOLD:
                    yawn_flag = False

                if YAWN_COUNT >= YAWN_ALERT_THRESHOLD:
                    shared_alert_list.append("YAWN")
                    last_alert_time = current_time
                    last_alert_message = "YAWN"
                    YAWN_COUNT = 0

        # 中央顯示大字警示（持續幾秒）
        if current_time - last_alert_time < ALERT_DISPLAY_DURATION:
            text_size = cv2.getTextSize(last_alert_message, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = 50
            cv2.putText(frame, last_alert_message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        # 顯示打哈欠次數
        cv2.putText(frame, f"Yawns: {YAWN_COUNT}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        try:
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.getWindowProperty("Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] OpenCV 視窗已關閉，結束程式")
                break
        except Exception as e:
            print(f"[ERROR] 顯示畫面時出錯: {e}")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 使用者按下 q 鍵，退出程式")
            break
        elif key == 32:  # 空白鍵重置
            reset_state()

    cap.release()
    cv2.destroyAllWindows()
