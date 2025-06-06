
import cv2
from multiprocessing import Process, Manager
from fatigue_detection.drowsiness_detection_mediapipe import start_drowsiness_detection
from driver_risk_alert_system.lane_tracker_module import LaneTracker
from audio_player import (
    play_alert_sound,
    DROWSINESS_ALERT_SOUND,
    YAWN_ALERT_SOUND,
    MID_RISK_ALERT_SOUND,
    HIGH_RISK_ALERT_SOUND
)
import time
import threading

def run_lane_tracker(shared_alert_list):
    tracker = LaneTracker(shared_alert_list)
    tracker.run()

def monitor_alerts(shared_alert_list):
    played_recently = {}
    cooldown = 2  # seconds

    while True:
        now = time.time()
        while shared_alert_list:
            print(shared_alert_list)
            alert = shared_alert_list.pop(0)

            if alert in played_recently and now - played_recently[alert] < cooldown:
                continue

            if alert == "DROWSINESS":
                play_alert_sound(DROWSINESS_ALERT_SOUND)
            elif alert == "YAWN":
                play_alert_sound(YAWN_ALERT_SOUND)
            elif alert == "RISK:mid":
                play_alert_sound(MID_RISK_ALERT_SOUND)
            elif alert == "RISK:high":
                play_alert_sound(HIGH_RISK_ALERT_SOUND)

            played_recently[alert] = now

        time.sleep(3)

if __name__ == "__main__":
    manager = Manager()
    shared_alert_list = manager.list()

    monitor_thread = threading.Thread(target=monitor_alerts, args=(shared_alert_list,))
    monitor_thread.daemon = True
    monitor_thread.start()

    p1 = Process(target=start_drowsiness_detection, args=(shared_alert_list,))
    p1.start()

    time.sleep(2)
    p2 = Process(target=run_lane_tracker, args=(shared_alert_list,))
    p2.start()

    p1.join()
    p2.join()
