# recognition-api/api/services/face_recognition_core.py

import cv2
import numpy as np
import os
import base64
from scipy.spatial.distance import euclidean 

# --- 辅助函数：将Base64字符串解码为OpenCV图像 (NumPy array) ---
def base64_to_cv2_image(base64_string: str) -> np.ndarray:
    """将Base64编码的图片字符串解码为OpenCV图像格式 (NumPy array)。"""
    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError("图片解码失败，可能图片格式无效或数据损坏。")
        return img_np
    except Exception as e:
        print(f"解码Base64图片失败: {e}")
        return None

# --- VisionServiceWorker 类：封装所有视觉处理逻辑和模型加载 ---
class VisionServiceWorker:
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VisionServiceWorker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

            # --- 模型文件路径配置 (作为实例属性) ---
            # 从 script/ 向上走两层 (../../) 到 recognition-api/，再进入 dnn_models/
            self.MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../api', '..', 'dnn_models')

            # 人脸检测模型 (用于找到人脸框)
            self.FACE_DETECTOR_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector.pbtxt')
            self.FACE_DETECTOR_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector_uint8.pb')
            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.7 

            # 人脸特征提取模型 (用于识别“是谁”)
            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, 'nn4.small2.v1.t7') 
            self.FACE_RECOGNITION_THRESHOLD = 0.8  # 欧氏距离阈值，可调整 (越小越相似)

            # 模型实例 (初始化为 None，在 _load_all_models 中加载)
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None

            # --- 初始化时加载所有模型 ---
            self._load_all_models()

    def _load_all_models(self):
        """内部方法：加载所有深度学习模型。"""
        print("Services: 正在加载人脸相关深度学习模型...")
        try:
            self.FACE_DETECTOR_NET = cv2.dnn.readNet(self.FACE_DETECTOR_WEIGHTS_PATH, self.FACE_DETECTOR_PROTOTXT_PATH)
            print("Services: 人脸检测模型加载完成。")
            self.FACE_RECOGNITION_NET = cv2.dnn.readNetFromTorch(self.FACE_RECOGNITION_MODEL_PATH)
            print("Services: 人脸特征提取模型加载完成。")
        except Exception as e:
            print(f"错误: Services: 无法加载一个或多个人脸模型。详情: {e}")
            print(f"确保模型文件在 '{os.path.abspath(self.MODEL_DIR)}' 目录下。")
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None

    # --- 核心 OpenCV 算法方法：人体检测 (此版本不包含人体检测模型加载，此函数会返回空列表) ---
    def detect_persons(self, frame: np.ndarray) -> list:
        """
        在给定帧中检测人体。此函数当前版本仅作占位。
        """
        return [] 

    # --- 核心 OpenCV 算法方法：人脸检测 (用于找到人脸框) ---
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        在给定帧中检测人脸。
        Args:
            frame (np.ndarray): 输入的视频帧 (OpenCV图像格式)。
        Returns:
            list: 检测到的人脸列表，每个元素包含字典 (class_name, confidence, box_coords)。
        """
        if self.FACE_DETECTOR_NET is None: return []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        self.FACE_DETECTOR_NET.setInput(blob)
        detections = self.FACE_DETECTOR_NET.forward()

        detected_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.FACE_DETECTOR_CONFIDENCE_THRESHOLD: 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int") 

                detected_faces.append({
                    'class_name': 'face', 
                    'confidence': round(float(confidence), 2),
                    'box_coords': [int(startX), int(startY), int(endX), int(endY)] 
                })
        return detected_faces

    # --- 核心 OpenCV 算法方法：提取人脸特征 ---
    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        从裁剪出的人脸图像中提取人脸特征向量（嵌入）。
        Args:
            face_image (np.ndarray): 裁剪好的人脸图像。
        Returns:
            np.ndarray: 128维的人脸特征向量。如果模型未加载或特征提取失败，返回空数组。
        """
        if self.FACE_RECOGNITION_NET is None:
            return np.array([]) 

        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (96, 96)), 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.FACE_RECOGNITION_NET.setInput(face_blob)
        features = self.FACE_RECOGNITION_NET.forward()
        return features.flatten() 

    # --- 核心 OpenCV 算法方法：比对人脸特征并识别身份 ---
    def recognize_face_identity(self, frame: np.ndarray, known_faces_data: list) -> list:
        """
        在给定帧中进行人脸识别，判断人脸身份。
        Args:
            frame (np.ndarray): 原始视频帧。
            known_faces_data (list): 已知人脸数据列表，每个元素包含 {'person_id': ..., 'person_name': ..., 'face_feature': np.array}
                                     face_feature 是从数据库加载的人脸特征向量。
        Returns:
            list: 识别结果列表，每个元素包含 (box_coords, identity, confidence, distance)。
        """
        (h, w) = frame.shape[:2] # 获取当前帧的宽度和高度，用于裁剪人脸时避免越界

        detected_faces = self.detect_faces(frame) 

        recognition_results = []
        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] 

            if cropped_face.size == 0: 
                continue 

            face_embedding = self.extract_face_features(cropped_face) 

            if face_embedding.size == 0: 
                recognition_results.append({
                    'box_coords': face_info['box_coords'],
                    'identity': 'unknown_feature_error',
                    'confidence': face_info['confidence'],
                    'distance': -1
                })
                continue

            best_match_name = 'Stranger' 
            best_match_id = None 
            min_distance = float('inf') 

            # 与已知人脸库进行比对
            for known_face in known_faces_data:
                # 确保将列表形式的 face_embedding 转换为 NumPy 数组
                known_embedding = np.array(known_face['face_embedding'], dtype=np.float32) 

                if known_embedding.shape != face_embedding.shape:
                    print(f"警告: 特征向量维度不匹配。已知维度: {known_embedding.shape}, 新维度: {face_embedding.shape}")
                    continue

                distance = euclidean(face_embedding, known_embedding) 

                if distance < min_distance:
                    min_distance = distance
                    best_match_name = known_face['name'] 
                    best_match_id = known_face['id'] 

            # 根据阈值判断是否匹配 (距离越小越相似)
            if min_distance < self.FACE_RECOGNITION_THRESHOLD: 
                identity_info = best_match_name 
            else:
                identity_info = 'Stranger' 

            recognition_results.append({
                'box_coords': face_info['box_coords'], 'identity': identity_info,
                'confidence': face_info['confidence'], 
                'distance': round(min_distance, 2),
                'person_id': person_id_info 
            })

        return recognition_results

# 在 services 模块被导入时，自动创建 VisionServiceWorker 的一个实例
vision_service_worker = VisionServiceWorker()