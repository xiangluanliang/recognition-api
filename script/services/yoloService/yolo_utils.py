# 调用yolo检测人物
from ultralytics import YOLO
from utils_pose import get_center_point
pose_model = YOLO("yolov8l-pose.pt")

def detect_people(frame):
    results = pose_model(frame)
    kpts_list, centers = [], []
    for r in results:
        if r.keypoints is None:
            continue
        for kp in r.keypoints:
            pts = kp.xy[0].cpu().numpy()
            kpts_list.append(pts)
            centers.append(get_center_point(pts))
    return kpts_list, centers
