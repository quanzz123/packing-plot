import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from sklearn.neighbors import NearestNeighbors
from collections import deque
import time
import os

videoPath = 'E:\\WORKSPACE\\packing-plot\\carPark.mp4'
modelPath = 'E:\\WORKSPACE\\packing-plot\\models\\yolo11_finetune.pt'
gatePos = (50, 360)

viTri = 'E:\\WORKSPACE\\packing-plot\\xac_dinh_vi_tri\\viTri'

confidence = 0.5
iou_threshold = 0.3  # Sửa tên biến: iuo -> iou_threshold
class_ids = [0]  # Sửa tên biến: clasID -> class_ids

width = 107
height = 48

VT = []
VT_history = {}

model = None


def khoiTao():
    global VT, model, VT_history
    try:
        model = YOLO(modelPath)
        print('Đã load YOLO11 finetune')
    except Exception as e:
        print(f'Lỗi khi tải model: {e}')
        return False

    if not load_VT():  # THIẾU "not"
        print('Không thể tải các vị trí đánh dấu')
        return False
    return True


def load_VT():
    """Tải các vị trí đỗ xe từ file có s�ẵn"""
    global VT, VT_history
    try:
        with open(viTri, 'rb') as f:
            VT = pickle.load(f)
        print(f'Đã tải {len(VT)} vị trí đỗ xe')

        # Khởi tạo lịch sử cho mỗi chỗ đỗ
        for i in range(len(VT)):
            VT_history[i] = deque([False] * 5, maxlen=5)
        return True

    except Exception as e:
        print(f'Không tìm thấy file vị trí: {e}')
        return False


def kiemTraVT():
    global VT

    if not VT:
        print('Danh sách vị trí trống')
        return False

    # Kiểm tra từng vị trí
    for i, vt in enumerate(VT):
        if not isinstance(vt, (tuple, list)) or len(vt) != 2:
            print(f'Vị trí {i} phải là tuple/list có 2 phần tử')
            return False
        if not all(isinstance(toaDo, (int, float)) for toaDo in vt):
            print(f'Vị trí {i} sai kiểu dữ liệu')
            return False

    print(f'✅ Đã kiểm tra {len(VT)} chỗ đỗ')
    return True


def phatHienXe(frame):
    """Phát hiện xe sử dụng YOLO"""
    global model
    ketQua = model(frame, conf=confidence, iou=iou_threshold, classes=class_ids,
                   verbose=False)  # SỬA: cof -> conf, iuo -> iou_threshold

    cars = []

    for kq in ketQua:
        boxes = kq.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0])  # SỬA: id -> class_id

            cars.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': class_id,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            })
    return cars


def tinhToanVungGiao(box1, box2):
    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2

    x1 = max(b1x1, b2x1)
    y1 = max(b1y1, b2y1)
    x2 = min(b1x2, b2x2)
    y2 = min(b1y2, b2y2)

    w_giao = max(0, x2 - x1)
    h_giao = max(0, y2 - y1)
    dienTichGiao = w_giao * h_giao

    b1_area = (b1x2 - b1x1) * (b1y2 - b1y1)
    b2_area = (b2x2 - b2x1) * (b2y2 - b2y1)

    dienTichHop = b1_area + b2_area - dienTichGiao

    # SỬA: Logic IoU đơn giản hơn
    return dienTichGiao / dienTichHop if dienTichHop > 0 else 0.0


def kiemTraChoCoBiChiem(vt, cars):
    vt_x, vt_y = vt
    vt_rect = (vt_x, vt_y, vt_x + width,
               vt_y + height)  # SỬA: (vt_x + height, vt_y + width) -> (vt_x + width, vt_y + height)

    for car in cars:
        car_box = car['bbox']
        iou = tinhToanVungGiao(vt_rect, car_box)

        if iou > 0.2:
            return True, car['conf']

    return False, 0.0


def update_spot_history(index, is_occupied):
    global VT_history
    VT_history[index].append(is_occupied)
    tiLe = sum(VT_history[index]) / len(
        VT_history[index])  # SỬA: sum(VT_history[index] / len(...)) -> sum(VT_history[index]) / len(...)

    return tiLe > 0.6


def timChodau(cho_dau):
    if not cho_dau:
        return None

    # SỬA HOÀN TOÀN hàm này
    empty_centers = np.array([(pos[0] + width // 2, pos[1] + height // 2) for pos in cho_dau])
    gate_array = np.array([gatePos])

    knn = NearestNeighbors(n_neighbors=min(3, len(empty_centers)))
    knn.fit(empty_centers)
    distances, indices = knn.kneighbors(gate_array)

    nearest_spots = []
    for j, i in enumerate(indices[0]):
        spot = cho_dau[i]
        distance = distances[0][j]
        nearest_spots.append((spot, distance))

    return nearest_spots


def draw_packing_infor(frame, empty_spots, occupied_spots, car_count, nearest_spot=None):
    total_spots = len(VT)
    empty_count = len(empty_spots)

    # SỬA: Font và màu sắc
    cv2.putText(frame, f'CHO TRONG: {empty_count}/{total_spots}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'XE PHAT HIEN: {car_count}',
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if nearest_spot:
        spot_pos, distance = nearest_spot
        cv2.putText(frame, f'CHO TRONG GAN NHAT: {distance:.1f}px',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, 'BAI DA DAY!',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.circle(frame, gatePos, 10, (0, 255, 255), -1)
    cv2.putText(frame, 'CUA RA VAO',
                (gatePos[0] + 15, gatePos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def draw_packing_spot(frame, empty_spots, occupied_spots, cars):
    for i, spot_pos in enumerate(VT):
        if spot_pos in occupied_spots:
            color = (0, 0, 255)  # Đỏ
            status = 'CO XE'
        else:
            color = (0, 255, 0)  # Xanh
            status = 'TRONG'

        # Vẽ khung chỗ đỗ
        cv2.rectangle(frame, spot_pos,
                      (spot_pos[0] + width, spot_pos[1] + height),
                      color, 2)

        # Vẽ số thứ tự
        cv2.putText(frame, f'{i + 1}',
                    (spot_pos[0] + 5, spot_pos[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Vẽ trạng thái
        cv2.putText(frame, status,
                    (spot_pos[0] + 5, spot_pos[1] + height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Vẽ bounding box xe
    for car in cars:
        x1, y1, x2, y2 = car['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Xe: {car["conf"]:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame


def hightlight_nearest_spot(frame, nearest_spot):
    if nearest_spot:
        spot_pos, distance = nearest_spot[0]  # Lấy chỗ gần nhất
        cv2.rectangle(frame, spot_pos,
                      (spot_pos[0] + width, spot_pos[1] + height),
                      (0, 255, 255), 4)  # Vàng đậm
        cv2.putText(frame, "GOI Y",
                    (spot_pos[0] + width // 2 - 20, spot_pos[1] + height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def analyze_parking_lot(frame):
    """Phân tích toàn bộ bãi đỗ xe"""
    cars = phatHienXe(frame)
    empty_spots = []
    occupied_spots = []

    for i, spot_pos in enumerate(VT):
        is_occupied, confidence = kiemTraChoCoBiChiem(spot_pos, cars)
        final_occupied = update_spot_history(i, is_occupied)

        if final_occupied:
            occupied_spots.append(spot_pos)
        else:
            empty_spots.append(spot_pos)

    return cars, empty_spots, occupied_spots


def process_frame(frame):
    """Xử lý một frame và trả về kết quả"""
    cars, empty_spots, occupied_spots = analyze_parking_lot(frame)
    nearest_spots = timChodau(empty_spots)

    result_frame = frame.copy()
    result_frame = draw_packing_spot(result_frame, empty_spots, occupied_spots, cars)

    if nearest_spots:
        result_frame = hightlight_nearest_spot(result_frame, nearest_spots)

    result_frame = draw_packing_infor(result_frame, empty_spots, occupied_spots, len(cars),
                                      nearest_spots[0] if nearest_spots else None)

    return result_frame, len(cars), len(empty_spots), len(occupied_spots)


def run_parking_system():
    """Hàm chính chạy hệ thống"""
    if not khoiTao():
        print("Khởi tạo hệ thống thất bại")
        return

    if not kiemTraVT():
        print("Vị trí đỗ xe không hợp lệ")
        return

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"Không thể mở video: {videoPath}")
        return

    print("\n🎮 Điều khiển:")
    print("  - Nhấn 'q' để thoát")
    print("  - Nhấn 'p' để tạm dừng")

    fps_counter = 0
    start_time = time.time()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Đã xử lý hết video")
                break

            processed_frame, vehicle_count, empty_count, occupied_count = process_frame(frame)

            # Hiển thị FPS
            fps_counter += 1
            if time.time() - start_time >= 1.0:
                fps = fps_counter / (time.time() - start_time)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                            (10, processed_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                fps_counter = 0
                start_time = time.time()

            cv2.imshow('He thong quan ly bai do xe', processed_frame)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Tạm dừng" if paused else "Tiếp tục")

    cap.release()
    cv2.destroyAllWindows()
    print("Đã thoát hệ thống")


if __name__ == "__main__":
    run_parking_system()