# recognition-api/api/services.py

import cv2
import numpy as np
import os
import base64
from scipy.spatial.distance import euclidean # 确保导入 euclidean，如果未安装，请 pip install scipy

# --- VisionService 类：封装所有视觉处理逻辑和模型加载 ---
class VisionService:
    _instance = None # 用于实现单例模式，确保只加载一次模型

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VisionService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # 确保初始化逻辑只运行一次 (针对单例模式)
        if not hasattr(self, '_initialized'):
            self._initialized = True

            # --- 模型文件路径配置 (作为实例属性，用 self. 前缀) ---
            self.MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'dnn_models')

            # MobileNet SSD 模型 (用于人体检测)
            self.PERSON_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'MobileNetSSD_deploy.prototxt')
            self.PERSON_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')
            self.PERSON_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                               "sofa", "train", "tvmonitor"]
            self.PERSON_CONFIDENCE_THRESHOLD = 0.6
            self.PERSON_COLORS = np.random.uniform(0, 255, size=(len(self.PERSON_CLASSES), 3))

            # 人脸检测模型 (用于找到人脸框)
            self.FACE_DETECTOR_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector.pbtxt')
            self.FACE_DETECTOR_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector_uint8.pb')
            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.7 

            # 人脸特征提取模型 (用于识别“是谁”)
            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, 'nn4.small2.v1.t7')
            self.FACE_RECOGNITION_THRESHOLD = 0.8  # 欧氏距离阈值，可调整 (越小越相似)

            # 模型实例 (初始化为 None，在 _load_all_models 中加载)
            self.PERSON_NET = None
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None

            # --- 初始化时加载所有模型 ---
            self._load_all_models()

    def _load_all_models(self):
        """内部方法：加载所有深度学习模型。"""
        print("Services: 正在加载所有深度学习模型...")
        try:
            # 加载人体检测模型
            self.PERSON_NET = cv2.dnn.readNetFromCaffe(self.PERSON_PROTOTXT_PATH, self.PERSON_WEIGHTS_PATH)
            print("Services: MobileNet SSD 人体检测模型加载完成。")

            # 加载人脸检测模型
            self.FACE_DETECTOR_NET = cv2.dnn.readNet(self.FACE_DETECTOR_WEIGHTS_PATH, self.FACE_DETECTOR_PROTOTXT_PATH)
            print("Services: 人脸检测模型加载完成。")

            # 加载人脸特征提取模型 (nn4.small2.v1.t7 是 PyTorch 模型)
            self.FACE_RECOGNITION_NET = cv2.dnn.readNetFromTorch(self.FACE_RECOGNITION_MODEL_PATH)
            print("Services: 人脸特征提取模型加载完成。")

        except Exception as e:
            print(f"错误: Services: 无法加载一个或多个模型。详情: {e}")
            print(f"确保模型文件在 '{os.path.abspath(self.MODEL_DIR)}' 目录下。") # 修正为 self.MODEL_DIR
            self.PERSON_NET = None
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None

    # --- 辅助函数：将Base64字符串解码为OpenCV图像 (NumPy array) ---
    @staticmethod # 这是一个静态方法，因为它不使用实例的属性
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

    # --- 核心 OpenCV 算法方法：人体检测 ---
    def detect_persons(self, frame: np.ndarray) -> list:
        """
        在给定帧中检测人体。
        Args:
            frame (np.ndarray): 输入的视频帧 (OpenCV图像格式)。
        Returns:
            list: 检测到的物体列表，每个元素包含字典 (class_name, confidence, box_coords)。
        """
        if self.PERSON_NET is None: return []

        (h, w) = frame.shape[:2]
        detected_objects = []

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5, swapRB=False, crop=False)

        self.PERSON_NET.setInput(blob)
        detections = self.PERSON_NET.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.PERSON_CONFIDENCE_THRESHOLD: # 使用 self.
                idx = int(detections[0, 0, i, 1])

                if self.PERSON_CLASSES[idx] == "person": # 使用 self.
                    class_name = self.PERSON_CLASSES[idx]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    detected_objects.append({
                        'class_id': idx, 'class_name': class_name,
                        'confidence': round(float(confidence), 2),
                        'box_coords': [int(startX), int(startY), int(endX), int(endY)]
                    })

        return detected_objects

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
            if confidence > self.FACE_DETECTOR_CONFIDENCE_THRESHOLD: # 使用 self.
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                detected_faces.append({
                    'class_name': 'face', 
                    'confidence': round(float(confidence), 2),
                    'box_coords': [int(startX), int(startY), int(endX), int(endY)]
                })
        return detected_faces

    # --- 新增核心 OpenCV 算法方法：提取人脸特征 ---
    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        从裁剪出的人脸图像中提取人脸特征向量（嵌入）。
        Args:
            face_image (np.ndarray): 裁剪好的人脸图像。
        Returns:
            np.ndarray: 128维的人脸特征向量。如果模型未加载或特征提取失败，返回空数组。
        """
        if self.FACE_RECOGNITION_NET is None:
            return np.array([]) # 模型未加载，返回空数组

        # OpenFace 模型期望 96x96 的 RGB 图像，并进行归一化
        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (96, 96)), 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.FACE_RECOGNITION_NET.setInput(face_blob)
        features = self.FACE_RECOGNITION_NET.forward()
        return features.flatten() # 将特征展平为1维向量 (例如128维)

    # --- 新增核心 OpenCV 算法方法：比对人脸特征并识别身份 ---
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
        # 获取当前帧的宽度和高度，用于裁剪人脸时避免越界
        (h, w) = frame.shape[:2] # <--- 添加这行

        detected_faces = self.detect_faces(frame) # 首先检测人脸

        recognition_results = []
        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            # 裁剪出人脸区域，确保裁剪范围有效且不越界
            cropped_face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] 

            if cropped_face.size == 0: # 如果裁剪区域为空，跳过
                continue 

            face_embedding = self.extract_face_features(cropped_face) # 提取特征

            if face_embedding.size == 0: # 如果特征提取失败 (例如模型未加载)
                recognition_results.append({
                    'box_coords': face_info['box_coords'],
                    'identity': 'unknown_feature_error',
                    'confidence': face_info['confidence'],
                    'distance': -1
                })
                continue

            best_match_id = 'unknown'
            best_match_name = '未知用户'
            min_distance = float('inf') # 初始化最小欧氏距离为无穷大

            # 与已知人脸库进行比对
            for known_face in known_faces_data:
                known_embedding = known_face['face_feature']
                if known_embedding.shape != face_embedding.shape:
                    print(f"警告: 特征向量维度不匹配。已知维度: {known_embedding.shape}, 新维度: {face_embedding.shape}")
                    continue

                distance = euclidean(face_embedding, known_embedding) # 计算欧氏距离

                if distance < min_distance:
                    min_distance = distance
                    best_match_id = known_face['person_id']
                    best_match_name = known_face['person_name']

            # 根据阈值判断是否匹配 (距离越小越相似)
            if min_distance < self.FACE_RECOGNITION_THRESHOLD: # 使用 self.FACE_RECOGNITION_THRESHOLD
                identity_info = best_match_name # 识别出是谁
            else:
                identity_info = '陌生人' # 距离太远，认为是陌生人

            recognition_results.append({
                'box_coords': face_info['box_coords'],
                'identity': identity_info,
                'confidence': face_info['confidence'], # 这是人脸检测的置信度
                'distance': round(min_distance, 2) # 识别的距离 (越小越好)
            })

        return recognition_results

# 在 services 模块被导入时，自动创建 VisionService 的一个实例
vision_service = VisionService()