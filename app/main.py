import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# =============================
# 🎯 CẤU HÌNH
# =============================
VIDEO_PATH = 'E:\\WORKSPACE\\packing-plot\\car_test.mp4'
MODEL_PATH = 'E:\\WORKSPACE\\packing-plot\\models\\yolo11_finetnune2class.pt'

# Class IDs
EMPTY_SPOT_CLASS = 0
OCCUPIED_SPOT_CLASS = 1

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.4

# Tham số OpenCV
MIN_SPOT_AREA = 800
MAX_SPOT_AREA = 15000
ASPECT_RATIO_RANGE = (1.2, 5.0)

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


def detect_spots_opencv(frame):
    """OpenCV detect chỗ đỗ tự động"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    opencv_spots = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if MIN_SPOT_AREA < area < MAX_SPOT_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if ASPECT_RATIO_RANGE[0] < aspect_ratio < ASPECT_RATIO_RANGE[1]:
                opencv_spots.append({
                    'bbox': (x, y, x + w, y + h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (x + w // 2, y + h // 2),
                    'method': 'opencv',
                    'contour': contour
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
        hist = cv2.calcHist([roi], [0], None, [64], [0, 256])
        hist_std = np.std(hist)

        occupancy_score = min(laplacian_var / 500 + hist_std / 50, 1.0)

        if occupancy_score > 0.3:
            occupied_spots.append(spot)
        else:
            empty_spots.append(spot)

    return empty_spots, occupied_spots


def merge_detections(yolo_empty, yolo_occupied, opencv_empty, opencv_occupied):
    """Kết hợp kết quả từ YOLO và OpenCV"""

    def get_spot_id(spot):
        x1, y1, x2, y2 = spot['bbox']
        return f"{x1 // 10}_{y1 // 10}_{x2 // 10}_{y2 // 10}"

    merged_empty = []
    merged_occupied = []
    processed_spots = {}

    for spot in yolo_empty + yolo_occupied:
        spot_id = get_spot_id(spot)
        if spot_id not in processed_spots:
            processed_spots[spot_id] = True
            if spot['class_id'] == EMPTY_SPOT_CLASS:
                merged_empty.append(spot)
            else:
                merged_occupied.append(spot)

    for spot in opencv_empty + opencv_occupied:
        spot_id = get_spot_id(spot)
        if spot_id not in processed_spots:
            processed_spots[spot_id] = True
            if spot in opencv_empty:
                merged_empty.append(spot)
            else:
                merged_occupied.append(spot)

    return merged_empty, merged_occupied


def update_spot_history(spots, is_empty=True):
    """Cập nhật lịch sử trạng thái spot"""
    for spot in spots:
        spot_id = f"{spot['bbox'][0]}_{spot['bbox'][1]}"

        if spot_id not in spot_history:
            spot_history[spot_id] = deque([is_empty] * 5, maxlen=5)
        else:
            spot_history[spot_id].append(is_empty)

        history_ratio = sum(spot_history[spot_id]) / len(spot_history[spot_id])
        spot['stable_status'] = history_ratio > 0.6

    return spots


def draw_hybrid_results(frame, yolo_spots, opencv_spots, merged_empty, merged_occupied, edges_frame):
    """Vẽ kết quả kết hợp - ĐÃ SỬA LỖI"""
    # Hiển thị edges detection
    edges_resized = cv2.resize(edges_frame, (320, 180))
    frame[10:190, 10:330] = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

    # Vẽ YOLO detections (màu cam)
    for spot in yolo_spots:
        x1, y1, x2, y2 = spot['bbox']
        color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "YOLO", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Vẽ OpenCV detections (màu tím)
    for spot in opencv_spots:
        x1, y1, x2, y2 = spot['bbox']
        color = (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, "CV", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Vẽ kết quả merged (màu đậm)
    for spot in merged_empty:
        x1, y1, x2, y2 = spot['bbox']
        color = (0, 255, 0) if spot.get('stable_status', True) else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, "TRONG", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for spot in merged_occupied:
        x1, y1, x2, y2 = spot['bbox']
        color = (0, 0, 255) if spot.get('stable_status', True) else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, "CO XE", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Thống kê - ĐÃ SỬA LỖI putText
    cv2.putText(frame, f"YOLO: {len(yolo_spots)} | OpenCV: {len(opencv_spots)}",
                (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"TRONG: {len(merged_empty)} | CO XE: {len(merged_occupied)}",
                (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "HYBRID SYSTEM: YOLO + OpenCV",
                (350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Chú thích màu - ĐÃ SỬA LỖI putText
    legend_y = 120
    cv2.putText(frame, "CHU THICH:", (350, legend_y),  # SỬA: Thêm tuple ()
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "YOLO Detection", (350, legend_y + 20),  # SỬA: Thêm tuple ()
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    cv2.putText(frame, "OpenCV Detection", (350, legend_y + 40),  # SỬA: Thêm tuple ()
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.putText(frame, "TRONG (Final)", (350, legend_y + 60),  # SỬA: Thêm tuple ()
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "CO XE (Final)", (350, legend_y + 80),  # SỬA: Thêm tuple ()
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame


def main_hybrid():
    """Hệ thống kết hợp YOLO + OpenCV"""
    if not initialize_system():
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    print("🚀 HYBRID SYSTEM: YOLO + OpenCV")
    print("🎯 Không cần file .pkl - Tự động hoàn toàn")

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

        # STABILIZATION
        merged_empty = update_spot_history(merged_empty, is_empty=True)
        merged_occupied = update_spot_history(merged_occupied, is_empty=False)

        # VISUALIZATION
        result_frame = draw_hybrid_results(
            frame, all_yolo_spots, opencv_spots,
            merged_empty, merged_occupied, edges_frame
        )

        # FPS
        if time.time() - start_time >= 1.0:
            fps = fps_counter / (time.time() - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.1f}",
                        (10, result_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            fps_counter = 0
            start_time = time.time()

        cv2.imshow('🚗 Hybrid Parking System (YOLO + OpenCV)', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_hybrid()