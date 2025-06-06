import os
import threading
import time
import pygame.mixer # [修改點 1: 導入 pygame.mixer]

# 獲取當前腳本的絕對路徑，用於構建音檔的相對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定義音檔路徑
AUDIO_CACHE_DIR = os.path.join(current_dir, "audio_cache")

# 確認 audio_cache 資料夾存在，如果不存在則創建
if not os.path.exists(AUDIO_CACHE_DIR):
    os.makedirs(AUDIO_CACHE_DIR)
    print(f"Created directory: {AUDIO_CACHE_DIR}")

# 定義所有警示音檔的完整路徑
DROWSINESS_ALERT_SOUND = os.path.join(AUDIO_CACHE_DIR, "drowsiness_alert.mp3")
YAWN_ALERT_SOUND = os.path.join(AUDIO_CACHE_DIR, "yawn_alert.mp3")
MID_RISK_ALERT_SOUND = os.path.join(AUDIO_CACHE_DIR, "mid_risk_alert.mp3")
HIGH_RISK_ALERT_SOUND = os.path.join(AUDIO_CACHE_DIR, "high_risk_alert.mp3")

# 用於防止同一音檔重複播放過快的字典
_last_play_time = {}
_PLAY_COOLDOWN = 2 # 秒，同一個音檔在兩秒內不會重複播放

# [修改點 2: 初始化 Pygame Mixer]
# Pygame Mixer 通常需要在程式啟動時初始化一次
# 由於 audio_player.py 可能被多個模組導入，我們在模組載入時進行初始化
try:
    # 設置音頻參數：頻率、大小（位元深度）、聲道數、緩衝大小
    # 44100 Hz 是 CD 品質音頻的標準採樣率
    # -16 代表 16 位元有符號音頻，負數表示小端序
    # 2 表示立體聲（1 表示單聲道）
    # 1024 是緩衝區大小（樣本數），越小延遲越低，但可能導致爆音
    pygame.mixer.init(44100, -16, 2, 1024)
    print("[INFO] Pygame mixer initialized.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Pygame mixer: {e}")
    print("Please ensure your audio drivers are working or try restarting your system.")


def play_alert_sound(sound_file_path):
    """
    播放指定的警示音檔，並加入冷卻時間機制避免過度頻繁播放。
    """
    global _last_play_time

    if not pygame.mixer.get_init():
        print("[WARNING] Pygame mixer not initialized, skipping sound playback.")
        return

    # 檢查音檔是否存在
    if not os.path.exists(sound_file_path):
        print(f"Warning: Sound file not found: {sound_file_path}")
        return

    # 檢查是否在冷卻時間內
    now = time.time()
    if sound_file_path in _last_play_time and (now - _last_play_time[sound_file_path] < _PLAY_COOLDOWN):
        print(f"Skipping playback of {os.path.basename(sound_file_path)} due to cooldown.")
        return

    _last_play_time[sound_file_path] = now

    # 使用獨立的執行緒播放音檔，避免阻塞主程式
    def _play():
        try:
            # [修改點 3: 使用 pygame.mixer.Sound 載入和播放]
            sound = pygame.mixer.Sound(sound_file_path)
            sound.play()
            # 可以根據需要等待音檔播放完成 (但不推薦在非阻塞情況下)
            # time.sleep(sound.get_length())
        except pygame.error as e: # 捕獲 pygame 相關錯誤
            print(f"Pygame Error playing sound {os.path.basename(sound_file_path)}: {e}")
        except Exception as e: # 捕獲其他未知錯誤
            print(f"Error playing sound {os.path.basename(sound_file_path)}: {e}")

    # 啟動新執行緒
    thread = threading.Thread(target=_play)
    thread.daemon = True # 將執行緒設為守護執行緒，這樣主程式結束時它會自動結束
    thread.start()

# 函式調用範例 (僅供測試，實際會在其他模組中調用)
if __name__ == "__main__":
    print("Testing audio_player.py with Pygame mixer...")
    # 確保你有這些音檔在 audio_cache 資料夾中進行測試
    # 如果你沒有，請先放入或自行創建一些短的 .mp3 檔案
    print("Playing drowsiness alert...")
    play_alert_sound(DROWSINESS_ALERT_SOUND)
    time.sleep(3) # 等待足夠時間讓音檔播放
    print("Playing mid risk alert...")
    play_alert_sound(MID_RISK_ALERT_SOUND)
    time.sleep(3) # 等待足夠時間讓音檔播放
    print("Playing high risk alert...")
    play_alert_sound(HIGH_RISK_ALERT_SOUND)
    time.sleep(3) # 等待足夠時間讓音檔播放
    print("Test complete.")