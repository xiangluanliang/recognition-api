import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from PIL import ImageFont, ImageDraw, Image

logging.getLogger('ultralytics').setLevel(logging.ERROR)
pose_model = YOLO("yolov8l-pose.pt")

POSE_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6)
]

fall_buffer = 5
fall_frames = set()
person_history = defaultdict(lambda: deque(maxlen=10))

def angle_between_points(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def draw_pose(frame, kpts, color=(0, 255, 0)):
    for point in kpts:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, color, -1)
    for i, j in POSE_PAIRS:
        if i < len(kpts) and j < len(kpts):
            pt1, pt2 = tuple(kpts[i]), tuple(kpts[j])
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

def get_center_point(kpts):
    return ((kpts[5][0] + kpts[6][0]) / 2, (kpts[5][1] + kpts[6][1]) / 2)

def match_person_id(current_centers, prev_centers, threshold=60):
    ids = []
    matched = set()
    for c in current_centers:
        found = False
        for pid, pc in prev_centers.items():
            if pid in matched:
                continue
            dist = np.linalg.norm(np.array(c) - np.array(pc))
            if dist < threshold:
                ids.append(pid)
                matched.add(pid)
                prev_centers[pid] = c
                found = True
                break
        if not found:
            new_id = max(prev_centers.keys(), default=0) + 1
            ids.append(new_id)
            prev_centers[new_id] = c
            matched.add(new_id)
    return ids

def detect_abnormal_behavior(video_path, output_path):
    print("🚀 开始处理视频...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "❌ 无法打开视频", None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    abnormal_count = 0
    prev_centers = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame)
        kpts_list, centers = [], []

        for r in results:
            if r.keypoints is None:
                continue
            for kp in r.keypoints:
                pts = kp.xy[0].cpu().numpy()
                kpts_list.append(pts)
                centers.append(get_center_point(pts))

        ids = match_person_id(centers, prev_centers)

        for i, kpts in enumerate(kpts_list):
            pid = ids[i]
            center = centers[i]
            mid_shoulder = (kpts[5] + kpts[6]) / 2
            mid_hip = (kpts[11] + kpts[12]) / 2
            mid_knee = (kpts[13] + kpts[14]) / 2
            angle = angle_between_points(mid_shoulder, mid_hip, mid_knee)
            nose_y = kpts[0][1]
            knee_y = min(kpts[13][1], kpts[14][1])
            height = np.linalg.norm(mid_shoulder - mid_knee)

            person_history[pid].append({
                'angle': angle,
                'center': center,
                'nose_y': nose_y,
                'knee_y': knee_y,
                'height': height,
                'frame': frame_idx
            })

            # --- 趋势判断 ---
            trend_fall = False
            if len(person_history[pid]) >= 3:
                a1 = person_history[pid][-2]['angle']
                a2 = angle
                angle_diff = abs(a2 - a1)

                c1 = np.array(person_history[pid][-2]['center'])
                c2 = np.array(center)
                move_dist = np.linalg.norm(c2 - c1)

                # 判断摔倒趋势（大幅度角度变化 + 位移快）
                trend_fall = (angle_diff > 30 and move_dist > 50)

            # --- 最终判断 ---
            is_fall = (angle < 85 and nose_y > knee_y - 20) or trend_fall
            label = "✅" if is_fall else "❌"

            print(f"[帧{frame_idx}] 角度: {angle:.2f}°, 鼻Y: {nose_y:.1f}, 膝Y: {knee_y:.1f}, 摔倒趋势: {'是' if trend_fall else '否'} → 判定: {label}")

            if is_fall:
                abnormal_count += 1
                for offset in range(-fall_buffer, fall_buffer + 1):
                    fall_frames.add(frame_idx + offset)

            color = (0, 0, 255) if frame_idx in fall_frames else (0, 255, 0)
            draw_pose(frame, kpts, color=color)

            x1, y1 = int(np.min(kpts[:, 0])), int(np.min(kpts[:, 1]))
            x2, y2 = int(np.max(kpts[:, 0])), int(np.max(kpts[:, 1]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{'Fall' if frame_idx in fall_frames else 'Walking'}  ID:{pid}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ 处理完成，共检测到疑似摔倒帧数: {abnormal_count}")
    return True, f"共检测到疑似摔倒帧数: {abnormal_count}", output_path
