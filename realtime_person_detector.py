# 这是用于本地开发和测试的脚本，用于演示OpenCV视觉模块如何从视频流中实时检测人体和人脸识别。
# 它会调用 api/services.py 中定义的 VisionService。

import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
from api.services import vision_service # 从 services 模块导入 vision_service 实例

# --- 配置 ---
STREAM_URL = 0 # 使用本地摄像头 (ID 0) 进行测试

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- 模拟已知人脸数据 ---
# 在实际项目中，这些数据将从数据库 (FaceInfo 模型) 中加载
# 这里的特征向量是随机生成的，仅用于测试代码流程，不会有实际识别效果
# 你需要替换成实际的人脸特征向量 (例如，你可以运行一次脚本，截取你的脸，提取特征后保存下来用于这里)
MOCKED_KNOWN_FACES_DATA = [
    {
        'person_id': 'user_001',
        'person_name': '张三',
        # 这是一个随机的128维向量，实际请替换为真实的特征向量
        'face_feature': np.random.rand(128).astype(np.float32) 
    },
    {
        'person_id': 'user_002',
        'person_name': '李四',
        'face_feature': np.random.rand(128).astype(np.float32) 
    }
]


# --- 1. 打开摄像头或视频文件 ---
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print(f"错误: 无法打开视频源: {STREAM_URL}")
    print("请检查摄像头连接或流媒体地址是否正确。")
    sys.exit(1)

print(f"视频流/摄像头已打开: {STREAM_URL}。按 'q' 键退出。")

# --- 2. 主处理循环 ---
while True:
    ret, frame = cap.read()

    if not ret:
        print("视频流读取失败或已结束。")
        break

    # --- 调用 VisionService 中的检测和识别函数 ---
    person_detections = vision_service.detect_persons(frame)
    # 调用人脸识别，传入当前帧和模拟的已知人脸数据
    face_recognition_results = vision_service.recognize_face_identity(frame, MOCKED_KNOWN_FACES_DATA)

    # --- 在原始帧上绘制检测结果 (可视化) ---
    # 绘制人体检测框 (紫色)
    for detection in person_detections:
        (startX, startY, endX, endY) = detection['box_coords']
        class_name = detection['class_name']
        confidence = detection['confidence']

        color = (128, 0, 128) # 紫色 (BGR格式)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        label_text = f"{class_name}: {confidence:.2f}"
        label_y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label_text, (startX, label_y), FONT, 0.5, color, 2)

    # 绘制人脸识别结果 (蓝色)
    for recognition_result in face_recognition_results:
        (startX, startY, endX, endY) = recognition_result['box_coords']
        identity = recognition_result['identity']
        confidence = recognition_result['confidence'] # 这是人脸检测的置信度
        distance = recognition_result['distance']     # 这是识别的距离

        color = (255, 0, 0) # 蓝色 (BGR格式)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # 显示识别结果和距离
        label_text = f"ID: {identity} Dist: {distance:.2f}"
        label_y_top = startY - 25 if startY - 25 > 25 else startY + 5 # 调整标签位置
        label_y_bottom = startY - 10 if startY - 10 > 10 else startY + 20

        cv2.putText(frame, label_text, (startX, label_y_top), FONT, 0.5, color, 2)
        cv2.putText(frame, f"Det Conf: {confidence:.2f}", (startX, label_y_bottom), FONT, 0.4, color, 1)

    # --- 显示处理后的帧 ---
    cv2.imshow("Real-time Human & Face Recognition (Demo)", frame)

    # --- 检测 'q' 键按下以退出 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- 资源清理 ---
cap.release()
cv2.destroyAllWindows()
print("程序已退出。")