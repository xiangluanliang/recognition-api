#主要函数，调用检测处理视频

import cv2
import os
from collections import defaultdict, deque
from yolo_utils import detect_people
from logic_tracker import match_person_id
from event_handlers import check_fall, save_clip, check_abnormal_distance, check_abnormal_overlap
from utils_pose import draw_pose
from constants import CLIP_DURATION_SECONDS
from api.models import EventLog, Camera
from django.utils import timezone

def draw_abnormal_zone(frame, coords):
    x1, y1, x2, y2 = coords
    return cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def detect_abnormal_behavior(video_path, output_path, abnormal_zone_coords, safe_distance):
    print("🚀 开始处理视频...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "❌ 无法打开视频", None

    camera = Camera.objects.first() or Camera.objects.create(name='Default Camera', is_active=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    abnormal_count = 0
    prev_centers = {}
    fall_clip_buffer = defaultdict(lambda: deque(maxlen=int(fps * CLIP_DURATION_SECONDS)))
    person_history = defaultdict(list)
    person_fall_status = defaultdict(lambda: {'fall_frame_count': 0, 'is_falling': False})
    person_intrusion_status = defaultdict(lambda: {'has_intruded': False})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = draw_abnormal_zone(frame, abnormal_zone_coords)
        kpts_list, centers = detect_people(frame)
        ids = match_person_id(centers, prev_centers)

        for i, center in enumerate(centers):
            pid = ids[i]
            fall_clip_buffer[pid].append(frame.copy())

        for i, kpts in enumerate(kpts_list):
            pid = ids[i]
            center = centers[i]
            is_fall, is_new_fall = check_fall(pid, kpts, center, frame_idx, person_history, person_fall_status)

            x1, y1 = int(kpts[:, 0].min()), int(kpts[:, 1].min())
            x2, y2 = int(kpts[:, 0].max()), int(kpts[:, 1].max())
            bbox = (x1, y1, x2, y2)

            is_intrusion = check_abnormal_overlap(bbox, abnormal_zone_coords)

            color = (0, 255, 0)
            if is_fall:
                color = (0, 0, 255)
                if is_new_fall:
                    abnormal_count += 1
                    clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'fall_clips', 'fall')
                    image_path = os.path.join("fall_clips", f"fall_{pid}_{frame_idx}.jpg")
                    cv2.imwrite(image_path, frame)
                    EventLog.objects.create(
                        event_type='person_fall',
                        camera=camera,
                        time=timezone.now(),
                        confidence=0.85,
                        image_path=image_path,
                        video_clip_path=clip_path,
                        person=None
                    )
            elif is_intrusion:
                color = (0, 0, 255)
                if not person_intrusion_status[pid]['has_intruded']:
                    person_intrusion_status[pid]['has_intruded'] = True
                    abnormal_count += 1
                    clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'intrusion_clips', 'intrusion')
                    image_path = os.path.join("intrusion_clips", f"intrusion_{pid}_{frame_idx}.jpg")
                    cv2.imwrite(image_path, frame)
                    EventLog.objects.create(
                        event_type='intrusion',
                        camera=camera,
                        time=timezone.now(),
                        confidence=0.95,
                        image_path=image_path,
                        video_clip_path=clip_path,
                        person=None
                    )
            else:
                person_intrusion_status[pid]['has_intruded'] = False

            draw_pose(frame, kpts, color)
            x1, y1 = int(kpts[:, 0].min()), int(kpts[:, 1].min())
            x2, y2 = int(kpts[:, 0].max()), int(kpts[:, 1].max())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{'Fall' if is_fall else 'Intrusion' if is_intrusion else 'Normal'} ID:{pid}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return True, f"共检测到异常事件: {abnormal_count}", output_path
