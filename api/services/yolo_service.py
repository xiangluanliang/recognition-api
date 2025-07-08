# api/services/yolo_service.py
import cv2
import tempfile
from ultralytics import YOLO
import os

#处理视频文件，通过YOLO经行检测

model = YOLO("yolov8n.pt")  # 模型可以换成你训练好的路径

def process_video(video_file):
    """
    接收 Django InMemoryUploadedFile，保存为临时视频文件，逐帧处理
    返回检测完的视频文件路径或帧列表
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        for chunk in video_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    results = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 推理
        result = model(frame)[0]  # 拿出第一个结果
        frame_with_boxes = result.plot()  # 画框
        # 保存帧（此处你也可以返回标记结果坐标）
        results.append(frame_with_boxes)
        frame_count += 1
        if frame_count > 10:  # 仅处理前10帧用于演示
            break

    cap.release()
    os.remove(tmp_path)

    return results  # 返回列表帧（供前端展示）
