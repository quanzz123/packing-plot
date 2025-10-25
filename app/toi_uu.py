import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# =============================
# 🎯 CẤU HÌNH
# =============================
VIDEO_PATH = 'E:\\WORKSPACE\\packing-plot\\carPark.mp4'
MODEL_PATH = 'E:\\WORKSPACE\\packing-plot\\models\\yolo11_finetnune2class.pt'

# Class IDs
EMPTY_SPOT_CLASS = 0  # "empty_spot"
OCCUPIED_SPOT_CLASS = 1  # "occupied_spot"

CONFIDENCE = 0.6
IOU_THRESHOLD = 0.5

# Biến toàn cục
model = None
spot_history = {}


def initialize_system():
    """Khởi tạo hệ thống"""
    global model
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Đã tải model YOLO")
        return True
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return False


def detect_spots_yolo(frame):
    """YOLO detect chỗ đỗ và trạng thái"""
    results = model(frame, conf=CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)

    yolo_empty = []
    yolo_occupied = []
    all_yolo_spots = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0])

            spot_info = {
                'bbox': (x1, y1, x2, y2),
                'class_id': class_id,
                'confidence': conf,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': (x2 - x1) * (y2 - y1),
                'method': 'yolo'
            }

            all_yolo_spots.append(spot_info)

            if class_id == EMPTY_SPOT_CLASS:
                yolo_empty.append(spot_info)
            elif class_id == OCCUPIED_SPOT_CLASS:
                yolo_occupied.append(spot_info)

    return yolo_empty, yolo_occupied, all_yolo_spots


def get_spot_color(class_id):
    """🎯 Chỉ trả về màu sắc - KHÔNG TEXT"""
    if class_id == EMPTY_SPOT_CLASS:
        return (0, 255, 0)  # 🟢 XANH - Trống
    elif class_id == OCCUPIED_SPOT_CLASS:
        return (0, 0, 255)  # 🔴 ĐỎ - Có xe
    else:
        return (255, 255, 0)  # 🟡 VÀNG - Không rõ


def draw_simple_results(frame, yolo_spots, merged_empty, merged_occupied):
    """Vẽ kết quả đơn giản - CHỈ MÀU SẮC"""

    # 🎯 VẼ KẾT QUẢ MERGED (chính thức)
    for spot in merged_empty + merged_occupied:
        x1, y1, x2, y2 = spot['bbox']

        # 🎯 Lấy màu dựa trên class_id
        if 'class_id' in spot:
            color = get_spot_color(spot['class_id'])
        else:
            # Mặc định dựa trên danh sách
            color = (0, 255, 0) if spot in merged_empty else (0, 0, 255)

        # Vẽ box với màu sắc
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # 🎯 VẼ SỐ THỨ TỰ (tùy chọn)
        spot_id = f"{x1 // 20}_{y1 // 20}"
        cv2.putText(frame, "#", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 🎯 THỐNG KÊ ĐƠN GIẢN
    total_empty = len(merged_empty)
    total_occupied = len(merged_occupied)
    total_spots = total_empty + total_occupied

    # Background cho thống kê
    cv2.rectangle(frame, (5, 5), (300, 90), (0, 0, 0), -1)

    cv2.putText(frame, f"TONG CHO: {total_spots}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 🟢 Trống - màu xanh
    cv2.putText(frame, f"TRONG: {total_empty}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🔴 Có xe - màu đỏ
    cv2.putText(frame, f"CO XE: {total_occupied}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def detect_spots_opencv(frame):
    """OpenCV detect chỗ đỗ tự động"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    opencv_spots = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if 800 < area < 15000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if 1.2 < aspect_ratio < 5.0:
                opencv_spots.append({
                    'bbox': (x, y, x + w, y + h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (x + w // 2, y + h // 2),
                    'method': 'opencv'
                })

    return opencv_spots, edges


def analyze_spot_status_opencv(frame, spots):
    """Phân tích trạng thái chỗ đỗ bằng OpenCV"""
    empty_spots = []
    occupied_spots = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for spot in spots:
        x1, y1, x2, y2 = spot['bbox']
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        occupancy_score = min(laplacian_var / 500, 1.0)

        if occupancy_score > 0.3:
            spot['class_id'] = OCCUPIED_SPOT_CLASS
            occupied_spots.append(spot)
        else:
            spot['class_id'] = EMPTY_SPOT_CLASS
            empty_spots.append(spot)

    return empty_spots, occupied_spots


def merge_detections(yolo_empty, yolo_occupied, opencv_empty, opencv_occupied):
    """Kết hợp kết quả từ YOLO và OpenCV"""

    def get_spot_id(spot):
        x1, y1, x2, y2 = spot['bbox']
        return f"{x1 // 15}_{y1 // 15}"

    merged_empty = []
    merged_occupied = []
    processed_spots = {}

    # Ưu tiên YOLO DETECTION
    for spot in yolo_empty + yolo_occupied:
        spot_id = get_spot_id(spot)
        if spot_id not in processed_spots:
            processed_spots[spot_id] = True
            if spot['class_id'] == EMPTY_SPOT_CLASS:
                merged_empty.append(spot)
            else:
                merged_occupied.append(spot)

    # Bổ sung OPENCV
    for spot in opencv_empty + opencv_occupied:
        spot_id = get_spot_id(spot)
        if spot_id not in processed_spots:
            processed_spots[spot_id] = True
            if 'class_id' in spot and spot['class_id'] == EMPTY_SPOT_CLASS:
                merged_empty.append(spot)
            elif 'class_id' in spot and spot['class_id'] == OCCUPIED_SPOT_CLASS:
                merged_occupied.append(spot)
            else:
                if spot in opencv_empty:
                    spot['class_id'] = EMPTY_SPOT_CLASS
                    merged_empty.append(spot)
                else:
                    spot['class_id'] = OCCUPIED_SPOT_CLASS
                    merged_occupied.append(spot)

    return merged_empty, merged_occupied


def main_simple():
    """Hệ thống đơn giản - CHỈ MÀU SẮC"""
    if not initialize_system():
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    print("🚀 HỆ THỐNG ĐƠN GIẢN")
    print("🎨 Trống: 🟢 XANH | Có xe: 🔴 ĐỎ")
    print("📊 Chỉ hiển thị thống kê, không text trên box")

    fps_counter = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter += 1

        # YOLO DETECTION
        yolo_empty, yolo_occupied, all_yolo_spots = detect_spots_yolo(frame)

        # OPENCV DETECTION
        opencv_spots, edges_frame = detect_spots_opencv(frame)
        opencv_empty, opencv_occupied = analyze_spot_status_opencv(frame, opencv_spots)

        # MERGE RESULTS
        merged_empty, merged_occupied = merge_detections(
            yolo_empty, yolo_occupied, opencv_empty, opencv_occupied
        )

        # 🎯 VẼ KẾT QUẢ ĐƠN GIẢN
        result_frame = draw_simple_results(frame, all_yolo_spots, merged_empty, merged_occupied)

        # FPS
        if time.time() - start_time >= 1.0:
            fps = fps_counter / (time.time() - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.1f}",
                        (10, result_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            fps_counter = 0
            start_time = time.time()

        cv2.imshow('🚗 Simple Parking System', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_simple()