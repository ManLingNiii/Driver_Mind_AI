import cv2 # OpenCV 影像處理
import numpy as np # 數值處理
import sys # 讀取命令列參數
from collections import deque # 雙端佇列，保存歷史線條資訊

#赫夫轉換

"""
儲存左右線歷史資料（最多 5 幀）
建立兩個「固定長度的雙端佇列（deque）」，每個最多只能存 5 條資料
為何要儲存5個歷史座標資料？ 
>> 因為每一幀偵測出來的車道線都可能會有一點點晃動，為了要讓車道線平滑一點，故儲存最近5幀的線條＆平均其位置
"""
left_line_history = deque(maxlen=15)
right_line_history = deque(maxlen=15)

def smooth_line(history, new_line):
    """
    平滑化車道線，讓車道線不抖動（用最近幾幀平均）
    history >> deque
    new_line >> 這一幀剛剛偵測到的線條，格式是 NumPy 陣列，如 [x1, y1, x2, y2]
    """
    if new_line is not None: #如果 new_line 存在，就把它加入 history(deque舊的會被刪掉)
        history.append(new_line)
    if not history:  # 目前 history 裡還沒有任何資料，就沒辦法計算平均，只能回傳 None(程式剛啟動時)
        return None
    avg_line = np.mean(history, axis=0).astype(int) #對 history 中所有的線條做「逐欄位平均」，也就是把 x1, y1, x2, y2 分別平均
    return avg_line

def detect_edges(frame):
    """
    接收一張影像 frame（通常是 BGR 彩色格式）
    目的：將影像轉換成「二值邊緣圖」，只保留邊界輪廓
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 將彩色影像 frame 轉換成灰階影像 gray(邊緣偵測只需要明暗變化，不需要顏色資訊)

    # 使用高斯模糊（Gaussian Blur）濾波影像，(5, 5) 是濾波核大小 (用意是去除雜訊，防止邊緣偵測被小細節干擾)
    # 0 表示讓 OpenCV 自動根據核大小決定標準差（σ）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 邊緣偵測演算法，找出邊界 	50 是低閾值（弱邊緣起始值），150 是高閾值（強邊緣保留值）
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(img):
    """
    定義車道區域（只關注下半部）
    """
    # 取得影像的高度與寬度。img 是一張灰階影像（從 detect_edges() 回傳的邊緣圖）
    height, width = img.shape

    # 建立一個與原圖 相同大小、全為 0（黑色）的遮罩圖像，準備在這張圖上畫出我們關注的區域
    mask = np.zeros_like(img)

    # 定義一個四邊形區域的頂點座標（以一個多邊形 polygon 表示），用來包住車道的形狀。這裡畫的是一個梯形區域，以畫面下方中間為重點。
    polygon = np.array([[
        (0, height), # 左下角
        (int(width * 0.4), int(height * 0.55)), # 左上角（靠中間）
        (int(width * 0.6), int(height * 0.55)), # 右上角（靠中間）
        (width, height) # 右下角
    ]])

    # 用 OpenCV 函數 fillPoly() 在 mask 上畫出這個多邊形，並將它塗成白色（255）。其餘區域仍是黑色（0）。
    cv2.fillPoly(mask, polygon, 255)

    # 對兩張圖片做「逐像素的 AND 運算」，用來保留同時為非零的區域。
    # 保留「既在邊緣圖像上是白色，又在 mask 中是 ROI 的區域」
    masked = cv2.bitwise_and(img, mask)
    return masked # 回傳處理後的結果，即只保留關注區域的圖像（其餘區域全為黑）

def detect_lines(edges):
    """
    HoughLinesP 找出所有候選線條 + 斜率篩選
    從經過邊緣偵測（Canny）後的影像中找出「直線」，並進行斜率篩選，只留下我們關心的線段（如斜向的車道線）
    """

    """
    使用 HoughLinesP（概率霍夫變換） 偵測影像中的所有可能線段
    此方法適合找直線，常用於偵測車道線、路邊線等。回傳的是一個 list，每一項是 [[x1, y1, x2, y2]] 的 numpy 陣列，代表一條線段。
    """
    raw_lines = cv2.HoughLinesP(edges, # 邊緣影像（二值圖）
                                1, # ρ (rho)：霍夫空間的距離解析度（1 像素）
                                np.pi / 180, # θ (theta)：角度解析度，這裡設為 1 度（以弧度為單位）
                                threshold=50, # 閾值：票數（累積的投票數）超過這個值才視為一條線
                                minLineLength=80, # 線段最小長度（像素）：太短的不算線
                                maxLineGap=60) # 同一條線容許的中斷間距：小間斷視為同一條線

    if raw_lines is None:
        return [] # 如果找不到任何線段（raw_lines 為 None），就直接回傳空列表，避免後續報錯。

    # 建立一個空列表，用來存放通過斜率條件篩選後的有效線段。
    filtered = []

    # 逐條遍歷偵測出來的線段，每一條線段由起點 (x1, y1) 和終點 (x2, y2) 表示。
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]

        # 這條線是垂直線（因為 x 不變，代表斜率為無限大），容易造成除以 0 的錯誤，或者在應用中不需要垂直線，所以直接跳過。
        if x2 == x1:
            continue

        # 計算這條線的斜率。slope = Δy / Δx。這能幫助我們過濾掉太水平或太垂直的線條。
        # 排除太水平的線（|slope| < 0.3）
        # 排除太垂直的線（|slope| > 5）
        slope = (y2 - y1) / (x2 - x1)
        if 0.3 < abs(slope) < 5:
            filtered.append(line)

    return filtered # 回傳所有符合條件的車道候選線段(清單)。

def average_slope_intercept(lines):
    """
    用最小平方擬合左右線
    定義一個函數 average_slope_intercept()，參數 lines 是先前用 cv2.HoughLinesP 偵測出來的多條線段（直線）
    """

    # 建立兩個清單：用來分別存放左邊與右邊的線條斜率與截距。會根據斜率正負分配。
    left_lines = []
    right_lines = []

    # 逐一取出每一條線（注意 line 是一個 list 包住 [x1, y1, x2, y2]），把座標拆解出來
    for line in lines:
        x1, y1, x2, y2 = line[0]

        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    left_avg = np.average(left_lines, axis=0) if left_lines else None
    right_avg = np.average(right_lines, axis=0) if right_lines else None
    return left_avg, right_avg

def fit_quadratic_curve(lines, side="left"):
    x_points, y_points = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if (side == "left" and slope < 0) or (side == "right" and slope > 0):
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])
    if len(x_points) < 2:
        return None
    degree = 2 if len(x_points) >= 6 else 1
    coeffs = np.polyfit(y_points, x_points, degree)
    return np.poly1d(coeffs)

def draw_curve_line(frame, poly_fn, color=(255, 255, 0)):
    if poly_fn is None:
        return
    h = frame.shape[0]
    y_vals = np.linspace(int(h * 0.6), h, 100)
    x_vals = poly_fn(y_vals)
    points = np.array([[int(x), int(y)] for x, y in zip(x_vals, y_vals)
                       if 0 <= x < frame.shape[1]], dtype=np.int32)
    for i in range(len(points) - 1):
        cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), color, 3)

def make_coordinates(frame, line_params):
    """
    根據直線參數計算端點座標
    """
    height, width, _ = frame.shape
    if line_params is None:
        return None
    slope, intercept = line_params
    y1 = height
    y2 = int(height * 0.6)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def draw_multicolor_lane(frame, left_line, right_line):
    """
    畫出紅橙綠三段風險區域，並保證回傳合法影像
    """
    if frame is None:
        print("[draw_multicolor_lane] 警告：輸入 frame 為 None")
        return np.zeros((720, 1280, 3), dtype=np.uint8)  # 根據預設解析度調整

    frame_copy = frame.copy()

    if left_line is None or right_line is None:
        print("[draw_multicolor_lane] 警告：缺少車道線，僅回傳原圖")
        return frame_copy

    try:
        lane_width = abs(left_line[0] - right_line[0])
        height, width = frame.shape[:2]
        y_bottom = height

        red_height = int(200)
        orange_height = 70
        green_height = 200

        red_y_top = y_bottom - red_height
        orange_y_top = red_y_top - orange_height
        left_y_min = min(left_line[1], left_line[3])
        right_y_min = min(right_line[1], right_line[3])
        max_top_y = max(left_y_min, right_y_min)
        green_y_top = max(orange_y_top - green_height, max_top_y)

        def interp_x(line, y):
            x1, y1, x2, y2 = line
            if y2 == y1:
                return x1
            return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

        left_x_red_bot = interp_x(left_line, y_bottom)
        left_x_red_top = interp_x(left_line, red_y_top)
        right_x_red_bot = interp_x(right_line, y_bottom)
        right_x_red_top = interp_x(right_line, red_y_top)
        left_x_orange_top = interp_x(left_line, orange_y_top)
        right_x_orange_top = interp_x(right_line, orange_y_top)
        left_x_green_top = interp_x(left_line, green_y_top)
        right_x_green_top = interp_x(right_line, green_y_top)

        red_zone = np.array([[left_x_red_bot, y_bottom], [left_x_red_top, red_y_top],
                             [right_x_red_top, red_y_top], [right_x_red_bot, y_bottom]])
        orange_zone = np.array([[left_x_red_top, red_y_top], [left_x_orange_top, orange_y_top],
                                [right_x_orange_top, orange_y_top], [right_x_red_top, red_y_top]])
        green_zone = np.array([[left_x_orange_top, orange_y_top], [left_x_green_top, green_y_top],
                               [right_x_green_top, green_y_top], [right_x_orange_top, orange_y_top]])

        overlay = np.zeros_like(frame_copy)
        cv2.fillPoly(overlay, [green_zone], (0, 255, 0))
        cv2.fillPoly(overlay, [orange_zone], (0, 165, 255))
        cv2.fillPoly(overlay, [red_zone], (0, 0, 255))

        red_width = abs(left_x_red_bot - right_x_red_bot)
        if red_width < lane_width * 0.4:
            cv2.putText(frame_copy, "WARNING: TOO CLOSE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return cv2.addWeighted(frame_copy, 1, overlay, 0.4, 1)

    except Exception as e:
        print("[❌ draw_multicolor_lane 畫圖失敗]", e)
        return frame_copy


def get_lane_roi_dynamic(left_line, right_line, frame_shape, speed=0, scale_factor=1.0):
    """
    使用左線、右線建立動態 ROI（高、中、低 + 左右側切入區），允許根據車速動態調整長度
    """
    if left_line is None or right_line is None:
        return {}

    height = frame_shape[0]
    width = frame_shape[1]
    y_bottom = height

    # 車道寬度與縮放
    left_x_bot = left_line[0]
    right_x_bot = right_line[0]
    lane_width = abs(right_x_bot - left_x_bot)

    base_scale = min(max(lane_width / 400, 0.5), 0.6)
    dynamic_scale = min(base_scale + speed * 0.005 * scale_factor, 1.0)
    scale = dynamic_scale

    red_height = int(150 * scale)
    orange_height = int(70 * scale)
    green_height = int(180 * scale)

    # 精準銜接：每一區 top 為上一區 top 減高
    red_y_top = y_bottom - red_height
    orange_y_bottom = red_y_top
    orange_y_top = orange_y_bottom - orange_height
    green_y_bottom = orange_y_top
    green_y_top = green_y_bottom - green_height

    # 預防太高超出畫面
    left_y_min = min(left_line[1], left_line[3])
    right_y_min = min(right_line[1], right_line[3])
    max_top_y = max(left_y_min, right_y_min)
    green_y_top = max(green_y_top, max_top_y)

    def interp_x(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return (x1 + x2) // 2
        return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    # 左右邊界點
    left_x_red_bot = interp_x(left_line, y_bottom)
    left_x_red_top = interp_x(left_line, red_y_top)
    left_x_orange_top = interp_x(left_line, orange_y_top)
    left_x_green_top = interp_x(left_line, green_y_top)

    right_x_red_bot = interp_x(right_line, y_bottom)
    right_x_red_top = interp_x(right_line, red_y_top)
    right_x_orange_top = interp_x(right_line, orange_y_top)
    right_x_green_top = interp_x(right_line, green_y_top)

    # 左右切入區延伸寬度（以車道寬延伸 0.9）
    side_offset = int(lane_width * 0.9)
    side_y_top = max(red_y_top - 100, 0)  # 上緣再往上拉長 80px
    right_x_side_bot = min(right_x_red_bot + side_offset, width - 1)
    right_x_side_top = min(right_x_red_top + side_offset, width - 1)
    left_x_side_bot = max(left_x_red_bot - side_offset, 0)
    left_x_side_top = max(left_x_red_top - side_offset, 0)

    roi_dict = {
        "high": np.array([
            [left_x_red_bot, y_bottom],
            [left_x_red_top, red_y_top],
            [right_x_red_top, red_y_top],
            [right_x_red_bot, y_bottom]
        ]),
        "mid": np.array([
            [left_x_red_top, red_y_top],
            [left_x_orange_top, orange_y_top],
            [right_x_orange_top, orange_y_top],
            [right_x_red_top, red_y_top]
        ]),
        "low": np.array([
            [left_x_orange_top, orange_y_top],
            [left_x_green_top, green_y_top],
            [right_x_green_top, green_y_top],
            [right_x_orange_top, orange_y_top]
        ]),
        "side_right": np.array([
            [right_x_red_bot, y_bottom],
            [right_x_red_top, red_y_top],
            [right_x_red_top + side_offset, side_y_top],
            [right_x_red_bot + side_offset, y_bottom]
        ]),
        "side_left": np.array([
            [left_x_red_bot - side_offset, y_bottom],
            [left_x_red_top - side_offset, side_y_top],
            [left_x_red_top, red_y_top],
            [left_x_red_bot, y_bottom]
        ])
    }

    return roi_dict, scale


def is_valid_lane_scene(left_line, right_line, frame_shape):
    """
    簡單判斷目前場景是否為有效的車道環境
    條件：
    1. 左右線皆存在
    2. 左右線之間的距離 > 最小寬度
    3. 車道線長度夠長（非碎線）
    """
    if left_line is None or right_line is None:
        return False

    height = frame_shape[0]
    min_lane_width = 100  # 兩條線之間最小距離
    min_line_height = int(height * 0.2)

    # 比較底部兩點的距離
    _, y1_left, _, y2_left = left_line
    _, y1_right, _, y2_right = right_line

    lane_width = abs(left_line[0] - right_line[0])
    left_line_length = abs(y2_left - y1_left)
    right_line_length = abs(y2_right - y1_right)

    if lane_width < min_lane_width:
        return False
    if left_line_length < min_line_height or right_line_length < min_line_height:
        return False

    return True

def process_frame(frame):
    """
    處理單幀主流程：包含直線與曲線擬合
    """
    global left_line_history, right_line_history

    try:
        # 邊緣偵測 + Hough 轉換
        edges = detect_edges(frame)
        roi = region_of_interest(edges)
        lines = detect_lines(roi)

        # ============ 線性擬合（備用） ============
        left_params, right_params = average_slope_intercept(lines)
        raw_left = make_coordinates(frame, left_params)
        raw_right = make_coordinates(frame, right_params)
        left_line = smooth_line(left_line_history, raw_left)
        right_line = smooth_line(right_line_history, raw_right)

        # ============ 曲線擬合 ============
        left_poly = fit_quadratic_curve(lines, side="left")
        right_poly = fit_quadratic_curve(lines, side="right")

        # ============ 畫主車道區域（紅橙綠） ============
        frame_with_colors = draw_multicolor_lane(frame, left_line, right_line)

        # ============ 畫曲線（若擬合成功） ============
        if left_poly is not None:
            draw_curve_line(frame_with_colors, left_poly, color=(0, 255, 255))  # 左側黃色線
        if right_poly is not None:
            draw_curve_line(frame_with_colors, right_poly, color=(0, 255, 255))  # 右側黃色線

        # ============ 動態 ROI 與風險切入區 ============
        roi_dict, scale = get_lane_roi_dynamic(left_line, right_line, frame.shape)
        scene_valid = is_valid_lane_scene(left_line, right_line, frame.shape)

        if "side_left" in roi_dict:
            overlay = frame_with_colors.copy()
            cv2.fillPoly(overlay, [roi_dict["side_left"]], (0, 140, 255))
            frame_with_colors = cv2.addWeighted(overlay, 0.4, frame_with_colors, 0.6, 0)

        if "side_right" in roi_dict:
            overlay = frame_with_colors.copy()
            cv2.fillPoly(overlay, [roi_dict["side_right"]], (0, 140, 255))
            frame_with_colors = cv2.addWeighted(overlay, 0.4, frame_with_colors, 0.6, 0)

        # ============ 錯誤處理備援 ============
        if frame_with_colors is None:
            print("[❌ process_frame] draw_multicolor_lane 回傳 None，回傳原圖")
            frame_with_colors = frame.copy()

        return frame_with_colors, roi_dict, scene_valid, left_line, right_line

    except Exception as e:
        print(f"[❌ process_frame Error] {e}")
        return frame.copy(), {}, False, None, None




if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "land2.mp4"  # 預設影片
    main(video_path)
