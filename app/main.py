import cv2
import numpy as np
import pickle
from tensorflow.python.util.numpy_compat import np_array
from torchgen.api.cpp import return_names
from ultralytics import YOLO
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque
import time
import  os

videoPath = 'Z:\\WORKSPACE\\bai_do_xe\\carPark.mp4'
modelPath = 'Z:\\WORKSPACE\\bai_do_xe\models\\yolo11_finetune.pt'
gatePos = (50,360)

viTri = 'Z:\\WORKSPACE\\bai_do_xe\\xac_dinh_vi_tri\\viTri'

confidence = 0.5
iuo = 0.3
clasID = [0]

width = 107
height = 48

VT = []
VT_history = {}

model = None

def khoiTao():
    global VT, model, VT_history
    try:
        model = YOLO(modelPath)
        print('đã load yolo 11 finetune')
    except:
        print('lỗi khi tải model')
        return  False
    if load_VT():
        print('không thể tại cáv vị trí đánh dấu')
        return False
    return  True

def load_VT():
    """Tải các vị trí đỗ xe từ file có sẵn"""
    global VT, VT_history
    try:
        with open(viTri, 'rb') as f:
            VT = pickle.load(f)
        print(f'đã tải {len(VT)} vị trí đỗ xe')

        #khởi tạo lịch sử cho mỗi chỗ đỗ
        for i in range(len(VT)):
            VT_history[i] = deque([False] * 5, maxlen=5)
        return True

    except:
        print('không tìm thấy file vị trí')

def kiemTraVT():
    global VT, VT_history

    if not VT:
        print('danh sách vị trí trống')
        return False
    #kiểm tra từng vị trí
    for i, vt in enumerate(VT):
        if not isinstance(vt, (tuple, list)) or len(vt) != 2:
            print(f'vi tri {i} phải là 2 phần tử ')
            return False
        if not all(isinstance(toaDo, (int,float)) for toaDo in vt):
            print(f'vị trí {i} sai kiểu dữ liệu')
            return False
    print(f'Đã kiểm tra {len(VT)} chỗ đỗ')
    return True

def phatHienXe(frame):
    """phát hiện xe sử dụng yolo"""
    global model
    ketQua = model(frame, cof = confidence, iou = iuo, classes = clasID)

    cars = []

    for kq in ketQua:
        boxes = kq.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            id = int(box.cls[0])

            cars.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'id': id,
                'center': ((x1+x2)//2, (y1+y2)//2)
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

    b1_area = (b1x2-b1x1)*(b1y2-b1y1)
    b2_area = (b2x2-b2x1)*(b2y2-b2y1)

    dienTichHop = b1_area + b2_area - dienTichGiao

    if dienTichGiao > dienTichHop:
        return 0
    else:
        iou =  dienTichGiao/dienTichHop
        return iou

def kiemTraChoCoBiChiem(vt, cars):
    vt_x, vt_y = vt
    vt_rect = (vt_x, vt_y, vt_x + height, vt_y + width)

    for car in cars:
        car_box = car['bbox']

        iou = tinhToanVungGiao(vt_rect, car_box)

        if iou > 0.2:
            return True, car['conf']

    return False, 0.0

def update_spot_history(index, is_occupied):
    global VT_history
    VT_history[index].append(is_occupied)
    tiLe = sum(VT_history[index] / len(VT_history[index]))

    if tiLe > 0.6:
        return True
    else :
        return False

def timChodau(cho_dau):
    if not cho_dau:
        return None

    for cho in cho_dau:
        center = np.array([cho[0] +width/2, cho[1] +height/2])
        gate_array = np_array([gatePos])

        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(center)
        distances, indices = knn.kneighbors(gate_array)

        nearest_spots= []
        for i, j in enumerate(indices[0]):
            spot = cho_dau[i]
            distance = distances[0][j]
            nearest_spots.append(spot,distance)

    return nearest_spots
def draw_packing_infor(frame, empty_spots, occupied_spots, car_count, nearest_spot= None):
    total_spots = len(VT)
    empty_count = len(empty_spots)

    cv2.putText(frame, f'cho trong {empty_count}/{total_spots}', (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    cv2.putText(frame, f'xe phat hien {car_count}', (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)

    if nearest_spot:
        spot_pos, distance = nearest_spot
        cv2.putText(frame, f'cho trong gan nha cach: {distance}', (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    else:
        cv2.putText(frame, 'bai da day', (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)

    cv2.circle(frame, gatePos, 10, (255,255,255), 2)
    cv2.putText(frame, 'cua ra vao', (gatePos[0] + 15, gatePos[1]), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    return frame

def draw_packing_spot(frame, empty_spots, occupied_spots, cars):
    for i, spot_pos in enumerate(VT):
        if spot_pos in occupied_spots:
            color = (0,0,255)
            sttus = 'co xe'
        else:
            color = (0,255,0)
            sttus = 'trong'
        cv2.rectangle(frame, spot_pos, (spot_pos[0] + width, spot_pos[1]+height ), (255,255,255), 2)
        cv2.putText(frame,f'i+1', (spot_pos[0]+5, spot_pos[1] +15), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
        cv2.putText(frame, (spot_pos[0]+5, spot_pos[1]+height-5), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    for car in cars:
        x1, y1, x2, y2 = car['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
        cv2.putText(frame, f'xe', (x1,y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)

    return frame


def hightlight_nearest_spot(frame, nearest_spot):
    if nearest_spot:
        spot_pos, distance = nearest_spot[0]
        cv2.rectangle(frame,spot_pos, (spot_pos[0] + width, spot_pos[1]), (255,255,255), 2)
        cv2.putText(frame, "GOI Y",
                    (spot_pos[0] + width // 2 - 20, spot_pos[1] + height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def analyze_parking_lot(frame):
    """Phân tích toàn bộ bãi đỗ xe - hàm chính"""
    global VT, VT_history

    # Phát hiện xe
    cars = phatHienXe(frame)

    empty_spots = []
    occupied_spots = []

    # Kiểm tra từng chỗ đỗ
    for i, spot_pos in enumerate(VT):
        is_occupied, confidence = kiemTraChoCoBiChiem(spot_pos,cars)
        final_occupied = update_spot_history(i, is_occupied)

        if final_occupied:
            occupied_spots.append(spot_pos)
        else:
            empty_spots.append(spot_pos)

    return cars, empty_spots, occupied_spots


def process_frame(frame):
    """Xử lý một frame và trả về kết quả"""
    # Phân tích bãi đỗ xe
    cars, empty_spots, occupied_spots = analyze_parking_lot(frame)

    # Tìm chỗ trống gần nhất
    nearest_spots = timChodau(empty_spots)

    # Vẽ kết quả lên frame
    result_frame = frame.copy()
    result_frame = draw_packing_spot(result_frame, empty_spots, occupied_spots, cars)

    if nearest_spots:
        result_frame = hightlight_nearest_spot(result_frame, nearest_spots)

    result_frame = draw_packing_infor(result_frame, empty_spots, occupied_spots, len(cars),
                                     nearest_spots[0] if nearest_spots else None)

    return result_frame, len(cars), len(empty_spots), len(occupied_spots)


def run_parking_system():
    """Hàm chính chạy hệ thống"""
    # Khởi tạo hệ thống
    if not khoiTao():
        print("❌ Khởi tạo hệ thống thất bại")
        return

    # Kiểm tra tính hợp lệ của vị trí đỗ xe
    if not kiemTraVT():
        print("❌ Vị trí đỗ xe không hợp lệ")
        return

    # Mở video
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {videoPath}")
        return

    print("\n🎮 Điều khiển:")
    print("  - Nhấn 'q' để thoát")
    print("  - Nhấn 'p' để tạm dừng")
    print("  - Nhấn 'i' để xem thông tin file")

    fps_counter = 0
    start_time = time.time()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ Đã xử lý hết video")
                break

            # Xử lý frame
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

            cv2.imshow('🚗 He thong quan ly bai do xe', processed_frame)

        # Xử lý phím
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸️ Tạm dừng" if paused else "▶️ Tiếp tục")
        elif key == ord('i'):
            print(f"\n📁 Thông tin file đang sử dụng:")
            print(f"   Đường dẫn: {viTri}")
            print(f"   Số chỗ đỗ: {len(VT)}")
            print(f"   Tồn tại: {os.path.exists(videoPath)}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã thoát hệ thống")