# recognition-api/api/services/send_face_recognition_data.py

import cv2
import numpy as np
import os
# 导入 VisionService 实例和 base64_to_cv2_image 辅助函数
# 注意导入路径：从当前包 (services) 向上找到 face_recognition_core 模块
from .face_recognition_core import vision_service, base64_to_cv2_image # <-- 修正了这一行！

# --- 核心处理函数 (你的 'send' 函数) ---
def process_face_recognition_task(image_base64_data: str, known_faces_data: list) -> dict:
    """
    接收Base64编码的图片数据，进行人脸识别，并判断用户是否合法。
    Args:
        image_base64_data (str): Base64编码的图片字符串。
        known_faces_data (list): 已知人脸数据列表，用于比对识别。
                                 每个元素包含 {'person_id': ..., 'person_name': ..., 'face_feature': np.array}。
    Returns:
        dict: 识别结果的汇总。
              例如: {'status': 'success', 'message': '识别完成', 'results': [...], 'is_legitimate_user': bool}
    """
    if not image_base64_data:
        return {'status': 'error', 'message': '未提供图片数据。'}

    # 1. 解码图片
    frame = base64_to_cv2_image(image_base64_data)
    if frame is None:
        return {'status': 'error', 'message': '图片解码失败。'}

    # 2. 调用 VisionService 进行人脸识别
    recognition_results = vision_service.recognize_face_identity(frame, known_faces_data)

    # 3. 判断是否有合法用户被识别 (这里是简化逻辑，未来会更复杂)
    is_legitimate_user = False
    recognized_identities = []
    for res in recognition_results:
        if res['identity'] != '陌生人' and res['identity'] != 'unknown_feature_error':
            is_legitimate_user = True
            recognized_identities.append(res['identity'])

    return {
        'status': 'success',
        'message': '人脸识别任务完成',
        'results': recognition_results,
        'is_legitimate_user': is_legitimate_user,
        'recognized_users': recognized_identities # 返回识别到的合法用户列表
    }