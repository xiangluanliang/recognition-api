# 异常事件检测函数
import os
import cv2
import numpy as np
from collections import deque
from django.utils import timezone
from utils_pose import angle_between_points
from api.models import EventLog

#异常距离检测
def check_abnormal_overlap(bbox, zone_coords):
    """
    判断人物框（bbox）是否与异常区域（zone_coords）重叠。
    bbox: (x1, y1, x2, y2)
    zone_coords: (zx1, zy1, zx2, zy2)
    """
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone_coords

    # 判断是否有交集
    inter_x1 = max(x1, zx1)
    inter_y1 = max(y1, zy1)
    inter_x2 = min(x2, zx2)
    inter_y2 = min(y2, zy2)

    # 有重叠区域
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

#摔倒情况检测
def check_fall(pid, kpts, center, frame_idx, person_history, person_fall_status, fall_window_size=3):
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

    trend_fall = False
    if len(person_history[pid]) >= 2:
        a1 = person_history[pid][-2]['angle']
        a2 = angle
        angle_diff = abs(a2 - a1)
        c1 = np.array(person_history[pid][-2]['center'])
        c2 = np.array(center)
        move_dist = np.linalg.norm(c2 - c1)
        trend_fall = (angle_diff > 30 and move_dist > 50)

    is_fall = (angle < 85 and nose_y > knee_y - 20) or trend_fall
    status = person_fall_status[pid]
    if is_fall:
        status['fall_frame_count'] += 1
        if status['fall_frame_count'] >= fall_window_size and not status['is_falling']:
            status['is_falling'] = True
            return True, True
        return True, False
    else:
        status['fall_frame_count'] = 0
        status['is_falling'] = False
        return False, False
#保存切片内容
def save_clip(pid, frame_idx, clip_buffer, fps, save_dir, event_type):
    os.makedirs(save_dir, exist_ok=True)
    clip_path = os.path.join(save_dir, f"{event_type}_{pid}_{frame_idx}.mp4")
    if not clip_buffer:
        return None
    height, width, _ = clip_buffer[0].shape
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in clip_buffer:
        writer.write(f)
    writer.release()
    return clip_path
