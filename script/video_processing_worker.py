# video_processing_worker.py

import cv2
import sys
import os
import numpy as np
import base64
import json
import psycopg2
import datetime
import time # 用于控制流处理的延迟和冷却时间
import requests # 用于发送HTTP请求
import threading # 用于异步发送警报

from flask import Flask, request, jsonify, Response
from scipy.spatial.distance import euclidean

# 配置 Flask 应用
app = Flask(__name__)

# 配置 Flask 日志以查看错误和警告
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (Worker) - %(message)s')


# --- VisionServiceWorker 类：封装了所有核心视觉和数据库逻辑 ---
class VisionServiceWorker:
    _instance = None # 用于实现单例模式

    def __new__(cls, *args, **kwargs):
        """实现单例模式的 __new__ 方法。确保 VisionServiceWorker 只有一个实例。"""
        if not cls._instance:
            cls._instance = super(VisionServiceWorker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """
        初始化模型、数据库连接参数并加载已知人脸数据。
        此方法被设计为在单例实例首次创建时只调用一次。
        """
        if not hasattr(self, '_initialized'): # 使用标志防止重复初始化
            self._initialized = True

            # --- 模型文件路径配置 ---
            current_script_dir = os.path.abspath(os.path.dirname(__file__))
            # === 这里是路径修正的地方 ===
            # 因为 video_processing_worker.py 在 'script' 文件夹内，而 dnn_models 在其上一级
            self.MODEL_DIR = os.path.join(current_script_dir, '..', 'dnn_models') # '..' 表示上一级目录
            # =========================

            self.FACE_DETECTOR_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector.pbtxt')
            self.FACE_DETECTOR_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector_uint8.pb')
            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.7 # 人脸检测的最低置信度阈值

            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, 'nn4.small2.v1.t7')
            self.FACE_RECOGNITION_THRESHOLD = 0.8  # 欧氏距离阈值，用于判断是否为同一个人（距离越小越相似）

            # 模型实例，初始化为 None，在 _load_all_models() 中加载
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None

            # --- 数据库连接参数 ---
            # 强烈建议通过环境变量配置这些敏感凭证。
            self.DB_HOST = os.environ.get('DB_HOST', 'your_db_host') # 例如 'localhost' 或云数据库 IP
            self.DB_NAME = os.environ.get('DB_NAME', 'your_db_name') # 例如 'rg_db'
            self.DB_USER = os.environ.get('DB_USER', 'your_db_user') # 例如 'rg_user'
            self.DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_db_password') # 数据库密码
            self.DB_PORT = os.environ.get('DB_PORT', '5432') # PostgreSQL 默认端口

            # 在 Worker 初始化时加载所有模型和已知人脸数据
            self._load_all_models()
            self.known_faces_data = self._load_known_faces()

    def _get_db_connection(self):
        """建立并返回一个 PostgreSQL 数据库连接。"""
        try:
            conn = psycopg2.connect(
                host=self.DB_HOST,
                database=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASSWORD,
                port=self.DB_PORT
            )
            return conn
        except Exception as e:
            app.logger.error(f"错误: 无法连接到数据库。详情: {e}")
            return None

    def _load_all_models(self):
        """加载人脸检测和人脸识别的深度学习模型。如果模型无法加载，则抛出 RuntimeError。"""
        app.logger.info("Worker: 正在加载人脸相关深度学习模型...")
        try:
            if not os.path.exists(self.FACE_DETECTOR_PROTOTXT_PATH):
                raise FileNotFoundError(f"人脸检测模型 prototxt 文件未找到: {self.FACE_DETECTOR_PROTOTXT_PATH}")
            if not os.path.exists(self.FACE_DETECTOR_WEIGHTS_PATH):
                raise FileNotFoundError(f"人脸检测模型权重文件未找到: {self.FACE_DETECTOR_WEIGHTS_PATH}")
            if not os.path.exists(self.FACE_RECOGNITION_MODEL_PATH):
                raise FileNotFoundError(f"人脸识别模型文件未找到: {self.FACE_RECOGNITION_MODEL_PATH}")

            self.FACE_DETECTOR_NET = cv2.dnn.readNet(
                self.FACE_DETECTOR_PROTOTXT_PATH,
                self.FACE_DETECTOR_WEIGHTS_PATH
            )
            app.logger.info("Worker: 人脸检测模型加载完成。")

            self.FACE_RECOGNITION_NET = cv2.dnn.readNetFromTorch(
                self.FACE_RECOGNITION_MODEL_PATH
            )
            app.logger.info("Worker: 人脸特征提取模型加载完成。")

        except Exception as e:
            app.logger.error(f"错误: Worker: 无法加载一个或多个人脸模型。详情: {e}")
            app.logger.error(f"请确保模型文件位于: '{os.path.abspath(self.MODEL_DIR)}' 目录下。")
            raise RuntimeError("人脸识别模型加载失败，请检查路径和文件。")

    def _load_known_faces(self) -> list:
        """从 PostgreSQL 的 'person' 表加载已知人脸数据（人员 ID、姓名、状态、人脸特征向量）。"""
        app.logger.info("Worker: 正在从数据库加载已知人脸数据...")
        known_faces_parsed = []
        conn = None
        try:
            conn = self._get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("SELECT id, name, state, face_image_path, face_embedding FROM person;")
                rows = cur.fetchall()
                cur.close()

                for row in rows:
                    person_id, name, state, face_image_path, face_embedding_obj = row
                    if face_embedding_obj is not None:
                        try:
                            # psycopg2 已经将 JSONB 类型自动转换为 Python 对象（如列表）
                            face_embedding = np.array(face_embedding_obj, dtype=np.float32)
                            known_faces_parsed.append({
                                'id': person_id,
                                'name': name,
                                'state': state,
                                'face_image_path': face_image_path,
                                'face_embedding': face_embedding
                            })
                        except Exception as e:
                            app.logger.warning(f"警告: 转换 '{name}' (ID: {person_id}) 的 face_embedding 为 NumPy 数组失败。详情: {e}")
                    else:
                        app.logger.warning(f"警告: 已知人脸数据 '{name}' (ID: {person_id}) 缺少 face_embedding。")

                app.logger.info(f"Worker: 已从数据库成功加载 {len(known_faces_parsed)} 条已知人脸数据。")
        except Exception as e:
            app.logger.error(f"错误: Worker: 无法从数据库检索已知人脸数据。详情: {e}")
            app.logger.warning("Worker: 将使用空的人脸数据列表，人脸识别功能将受到限制。")
        finally:
            if conn:
                conn.close()
        return known_faces_parsed

    @staticmethod
    def base64_to_cv2_image(base64_string: str) -> np.ndarray:
        """将 Base64 编码的字符串解码为 OpenCV 图像（NumPy 数组）。"""
        try:
            img_bytes = base64.b64decode(base64_string)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("图片解码失败。可能图片格式无效或数据损坏。")
            return img
        except Exception as e:
            app.logger.error(f"错误: 解码 Base64 图片失败: {e}")
            return None

    def detect_faces(self, frame: np.ndarray) -> list:
        """在图像帧中检测人脸。返回一个字典列表，每个字典包含人脸的边界框坐标和置信度。"""
        if self.FACE_DETECTOR_NET is None:
            app.logger.warning("警告: 人脸检测模型未加载。无法检测人脸。")
            return []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.FACE_DETECTOR_NET.setInput(blob)
        detections = self.FACE_DETECTOR_NET.forward()

        detected_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2] # 获取检测置信度
            if confidence > self.FACE_DETECTOR_CONFIDENCE_THRESHOLD:
                # 将边界框坐标缩放回原始帧尺寸
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                detected_faces.append({
                    'box_coords': [int(startX), int(startY), int(endX), int(endY)], # 确保是标准Python int
                    'confidence': float(confidence) # 确保是标准Python float
                })
        return detected_faces

    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        """从裁剪出的人脸图像中提取 128 维人脸特征向量。"""
        if self.FACE_RECOGNITION_NET is None:
            app.logger.warning("警告: 人脸特征提取模型未加载。无法提取特征。")
            return np.array([])

        # 对人脸图像进行预处理，以符合识别模型的输入要求（调整大小到 96x96，归一化像素）
        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (96, 96)), 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.FACE_RECOGNITION_NET.setInput(face_blob)
        features = self.FACE_RECOGNITION_NET.forward()
        return features.flatten() # 返回展平的 128 维向量

    def recognize_face_identity(self, frame: np.ndarray, known_faces_data: list) -> list:
        """
        在图像帧中检测人脸，提取其特征，并与已知人脸进行比对。
        返回每个检测到人脸的识别结果列表，包含人员状态。
        """
        results = []
        detected_faces = self.detect_faces(frame) # 使用 VisionServiceWorker 自身的人脸检测方法

        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            
            h, w = frame.shape[:2]
            # 裁剪人脸区域，确保坐标不超出图像边界
            startY, endY = max(0, y1), min(h, y2)
            startX, endX = max(0, x1), min(w, x2)

            cropped_face = frame[startY:endY, startX:endX]

            if cropped_face.size == 0:
                app.logger.warning("警告: 裁剪到的人脸区域为空。跳过特征提取。")
                results.append({
                    'box_coords': face_info['box_coords'],
                    'confidence': float(face_info['confidence']),
                    'identity': 'unknown_feature_error',
                    'distance': -1.0,
                    'person_id': None,
                    'person_state': None
                })
                continue

            face_embedding = self.extract_face_features(cropped_face)

            if face_embedding.size == 0:
                app.logger.warning("警告: 无法从人脸区域提取特征。")
                results.append({
                    'box_coords': face_info['box_coords'],
                    'confidence': float(face_info['confidence']),
                    'identity': 'unknown_feature_error',
                    'distance': -1.0,
                    'person_id': None,
                    'person_state': None
                })
                continue

            min_distance = float('inf')
            identity = 'Stranger'
            person_id = None
            person_state = None

            for known_face in known_faces_data:
                known_embedding = known_face['face_embedding']

                if known_embedding.shape != face_embedding.shape:
                    app.logger.warning(f"警告: 特征向量维度不匹配。已知维度: {known_embedding.shape}, 新维度: {face_embedding.shape}")
                    continue

                distance = euclidean(face_embedding, known_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    identity = known_face['name']
                    person_id = known_face['id']
                    person_state = known_face['state']

            # 判断最终身份和状态
            if min_distance < self.FACE_RECOGNITION_THRESHOLD:
                # 保持当前的 identity, person_id, person_state
                pass
            else:
                identity = 'Stranger'
                person_id = None
                person_state = None # 陌生人则状态为 None

            results.append({
                'box_coords': face_info['box_coords'],
                'confidence': float(face_info['confidence']),
                'identity': identity,
                'distance': float(min_distance),
                'person_id': person_id,
                'person_state': person_state
            })
        return results

    def _log_recognition_to_db(self, log_data: dict):
        """将人脸识别事件记录到 PostgreSQL 数据库的 'event_logs' 表中。"""
        conn = None
        try:
            conn = self._get_db_connection()
            if conn:
                cur = conn.cursor()
                insert_query = """
                INSERT INTO event_logs (event_type, time, confidence, image_path, video_clip_path, camera_id, person_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
                cur.execute(insert_query, (
                    log_data.get('event_type'),
                    datetime.datetime.now(datetime.timezone.utc), # 使用 UTC 时区
                    log_data.get('confidence'),
                    log_data.get('image_path', ''),
                    log_data.get('video_clip_path', ''),
                    log_data.get('camera_id'),
                    log_data.get('person_id')
                ))
                conn.commit()
                cur.close()
                app.logger.info(f"日志已成功写入数据库: 事件类型 '{log_data.get('event_type')}'，人员 ID: {log_data.get('person_id')}")
        except Exception as e:
            app.logger.error(f"错误: 写入事件日志到数据库失败。详情: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _save_new_person_to_db(self, person_name: str, state: int, face_image_path: str, face_embedding_list: list) -> int | None:
        """将新录入的人脸信息保存到 'person' 表中。返回新生成的 person_id 或 None。"""
        conn = None
        new_person_id = None
        try:
            conn = self._get_db_connection()
            if conn:
                cur = conn.cursor()
                face_embedding_json_str = json.dumps(face_embedding_list)
                insert_query = """
                INSERT INTO person (name, state, face_image_path, face_embedding)
                VALUES (%s, %s, %s, %s) RETURNING id;
                """
                cur.execute(insert_query, (person_name, state, face_image_path, face_embedding_json_str))
                new_person_id = cur.fetchone()[0]
                conn.commit()
                cur.close()
                app.logger.info(f"成功将新人员 '{person_name}' (ID: {new_person_id}, State: {state}) 保存到数据库。")
                return new_person_id
        except Exception as e:
            app.logger.error(f"错误: 无法将新人员信息保存到数据库。详情: {e}")
            app.logger.error("请检查数据库连接配置、表结构或是否存在违反唯一约束的情况。")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
        return None
    
    # --- 活体检测方法 (基于图像模糊度的简易实现) ---
    def perform_liveness_check(self, frame: np.ndarray, detected_faces: list) -> bool:
        """
        进行活体检测。这是一个简易的基于图像模糊度的实现。
        通过计算人脸区域的拉普拉斯方差来判断图像的清晰度。
        较低的方差可能表示图像模糊、平坦，可能是欺诈（例如照片、视频回放）。

        参数:
            frame (np.ndarray): 当前视频帧（彩色BGR图像）。
            detected_faces (list): 当前帧中检测到的人脸边界框信息列表，每个元素包含 'box_coords' 和 'confidence'。

        返回值:
            bool: True 表示判定为活体，False 表示判定为欺骗。
        """
        # 如果没有检测到人脸，默认通过活体检测，因为没有目标进行欺诈（或认为这是人脸检测器的职责）
        if not detected_faces:
            return True

        # 设置拉普拉斯方差阈值。这个值需要根据您的实际测试场景进行调整。
        # 较高的值会使检测更严格（更容易误报活体为欺诈），较低的值会更宽松。
        # 经验值：低于此阈值可能被认为是欺骗。
        # 这是一个需要根据实际测试环境和摄像头质量进行反复调整的参数。
        LIVENESS_BLUR_THRESHOLD = 100.0 # 示例值，建议根据实际情况测试调整

        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            
            # 确保人脸区域坐标在图像边界内
            h, w, _ = frame.shape
            startY, endY = max(0, y1), min(h, y2)
            startX, endX = max(0, x1), min(w, x2)

            # 裁剪人脸区域
            face_roi = frame[startY:endY, startX:endX]

            # 检查裁剪区域是否有效 (避免处理空或过小的区域)
            if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20: # 至少20x20像素
                app.logger.warning(f"活体检测: 人脸ROI过小或无效 ({face_roi.shape[1]}x{face_roi.shape[0]})，跳过模糊度检测。")
                continue # 跳过当前人脸，检查下一张

            # 将人脸区域转换为灰度图
            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # 计算拉普拉斯算子，然后计算方差
            # 拉普拉斯方差是衡量图像清晰度或焦距的标准指标
            laplacian_variance = cv2.Laplacian(gray_face_roi, cv2.CV_64F).var()

            app.logger.debug(f"人脸区域 ({startX},{startY})-({endX},{endY}) 拉普拉斯方差: {laplacian_variance:.2f}")

            # 如果拉普拉斯方差低于阈值，则认为可能是非活体（模糊或平坦）
            if laplacian_variance < LIVENESS_BLUR_THRESHOLD:
                app.logger.warning(f"警告: 检测到人脸区域模糊/平坦 (方差: {laplacian_variance:.2f} < 阈值: {LIVENESS_BLUR_THRESHOLD:.2f})，可能为欺骗攻击。")
                return False # 只要有一张人脸被判定为非活体，就返回 False

        # 如果所有检测到的人脸都通过了模糊度检测，则认为当前帧是活体
        return True


# --- VisionServiceWorker 单例实例 ---
# 在 Flask 应用启动时创建，加载模型和数据
vision_service_worker = VisionServiceWorker()


# --- 全局配置：主应用报警接收URL ---
# 替换为您的主 Django 应用或其他报警系统的实际接收URL
# 例如：'http://your_django_app_host:8000/api/alerts/receive/'
ALERT_RECEIVER_URL = os.environ.get('ALERT_RECEIVER_URL', 'http://127.0.0.1:8000/api/alerts/receive/')


# --- 辅助函数：发送警报到主应用 ---
def send_alert_to_main_app(alert_data: dict):
    """
    向主应用或其他报警系统发送 HTTP POST 请求。
    使用线程发送，避免阻塞视频流处理，提高响应速度。
    """
    import threading # 局部导入，避免循环依赖，因为 VisionServiceWorker 已经导入了requests

    if not ALERT_RECEIVER_URL:
        app.logger.warning("警告: 未配置 ALERT_RECEIVER_URL，无法发送报警通知。")
        return

    def _send_request():
        try:
            # 设置超时时间，避免长时间阻塞
            response = requests.post(ALERT_RECEIVER_URL, json=alert_data, timeout=5)
            response.raise_for_status() # 对非 2xx 响应（如 404, 500）抛出 HTTPError 异常
            app.logger.info(f"报警通知已成功发送到主应用: {alert_data['alert_type']}")
        except requests.exceptions.Timeout:
            app.logger.error(f"发送报警通知到主应用超时 (5秒): {ALERT_RECEIVER_URL}")
        except requests.exceptions.ConnectionError as e:
            app.logger.error(f"发送报警通知到主应用失败 (连接错误): {ALERT_RECEIVER_URL}, 详情: {e}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"发送报警通知到主应用失败 (HTTP请求错误): {e}")
        except Exception as e:
            app.logger.error(f"发送报警通知时发生未知错误: {e}")


    # 使用线程发送请求，不阻塞主视频流处理循环
    threading.Thread(target=_send_request).start()


# --- MJPEG 流生成函数：处理 RTMP 流，进行识别，并返回 JPEG 帧 ---
def gen_frames(stream_url: str, camera_id: int):
    """
    从 RTMP 流中拉取视频帧，进行人脸识别、绘制结果，并以 MJPEG HTTP 流形式返回。
    """
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        app.logger.error(f"错误: 无法打开视频流: {stream_url}")
        # 返回一个包含错误信息的帧，供前端显示
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8) # 黑色帧
        cv2.putText(error_frame, "STREAM OFFLINE / ERROR", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return # 无法打开流，退出函数

    # 帧处理和报警冷却的配置
    frame_skip = 5  # 每隔 N 帧进行一次人脸识别和日志记录，降低 CPU 负载
    frame_count = 0

    # 用于确保日志和警报只在首次检测到时触发，或每隔一段时间触发（冷却机制）
    # 存储 {人员ID/陌生人标识: 上次报警时间戳}
    last_alert_time = {
        'CRITICAL': {}, # 针对危险人物
        'WARNING': {},  # 针对陌生人 (使用 'Stranger' 作为 key)
        'INFO': {},     # 针对已知非危险人物
        'LIVENESS_FRAUD': 0 # 活体欺骗的冷却时间
    }
    LIVENESS_COOLDOWN_SECONDS = 10 # 活体欺骗报警的冷却时间（秒，可适当长一些）

    app.logger.info(f"成功打开视频流: {stream_url}")

    while True:
        success, frame = cap.read()
        if not success:
            app.logger.warning(f"警告: 无法从流 {stream_url} 读取帧。尝试重连...")
            cap.release() # 释放当前资源
            time.sleep(2) # 等待几秒后尝试重连
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                app.logger.error(f"错误: 流 {stream_url} 重连失败。退出流处理。")
                break # 重连失败，退出循环
            continue # 继续循环，等待下一帧

        current_time = time.time() # 获取当前时间戳

        # --- 活体检测和人脸识别逻辑 ---
        # 仅对每 frame_skip 帧进行人脸识别和日志记录，但所有帧都用于视频流输出
        if frame_count % frame_skip == 0:
            # 1. 人脸检测 (活体检测和识别都需要人脸框)
            detected_faces_in_frame = vision_service_worker.detect_faces(frame)

            # 2. 活体检测 (在人脸识别之前)
            is_live_person = vision_service_worker.perform_liveness_check(frame, detected_faces_in_frame)

            if not is_live_person:
                # 活体检测失败：记录欺骗告警并发送报警 (有冷却机制)
                if current_time - last_alert_time['LIVENESS_FRAUD'] > LIVENESS_COOLDOWN_SECONDS:
                    app.logger.warning(f"摄像头 {camera_id} 活体检测失败！疑似欺骗攻击。")
                    # 记录活体欺骗事件到数据库
                    vision_service_worker._log_recognition_to_db({
                        'event_type': 'LIVENESS_FRAUD_DETECTED',
                        'confidence': 1.0, # 活体检测告警通常置信度为1.0
                        'image_path': f'stream_frame_{frame_count}',
                        'camera_id': camera_id,
                        'person_id': None, # 欺骗攻击未识别到具体人员
                        'video_clip_path': f'live_stream_cam_{camera_id}',
                        'identity': 'Liveness_Fraud' # 特殊标识，方便日志查找
                    })
                    # 发送高优先级报警给主应用
                    send_alert_to_main_app({
                        "alert_type": "LIVENESS_FRAUD_DETECTED",
                        "alert_level": "CRITICAL",
                        "camera_id": camera_id,
                        "person_name": "欺骗攻击",
                        "person_id": None,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })
                    last_alert_time['LIVENESS_FRAUD'] = current_time # 更新冷却时间

                # 在视频流上显示欺骗警报，并跳过当前帧的人脸识别结果显示
                cv2.putText(frame, "【欺骗攻击】检测到非活体！", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # 红色警告字样

                # ⚠️ 跳过本帧后续的人脸识别和正常绘制，直接编码和yield
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                frame_count += 1
                continue # 继续到下一帧，不再进行本帧的常规人脸识别
            
            # 3. 如果活体检测通过，才进行人脸识别
            recognition_results = vision_service_worker.recognize_face_identity(frame, vision_service_worker.known_faces_data)
            
            # 用于在帧上显示和日志记录的临时变量
            current_frame_identities_on_screen = []
            has_dangerous_person_this_frame = False
            has_stranger_this_frame = False

            for res in recognition_results:
                x1, y1, x2, y2 = res['box_coords']
                # 确保坐标在图像范围内
                h, w, _ = frame.shape
                # 坐标顺序应为 (startX, startY, endX, endY)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2) 

                name_to_display = res['identity']
                color = (0, 255, 0) # 绿色：已知人员
                event_type = 'FACE_KNOWN_PERSON_DETECTED'
                # 构造视频片段路径，例如 'rtmp://your.rtmp.server.ip:9090/live/1'
                # 从 stream_url 提取主机部分，避免硬编码端口和协议
                stream_host_path = stream_url.split('//')[1].split('/')[0] if '//' in stream_url else stream_url.split('/')[0]
                log_video_clip_path = f'rtmp://{stream_host_path}/live/{camera_id}'

                alert_triggered_for_this_person_cooldown = False # 用于判断是否触发了冷却期外的报警

                # 根据识别结果确定显示名称、颜色、事件类型和是否触发外部报警
                if res['identity'] == 'Stranger':
                    name_to_display = "陌生人"
                    color = (0, 165, 255) # 橙色：陌生人
                    event_type = 'FACE_STRANGER_DETECTED'
                    has_stranger_this_frame = True
                    # 检查陌生人是否在冷却期
                    if 'Stranger' not in last_alert_time['WARNING'] or \
                       current_time - last_alert_time['WARNING']['Stranger'] > ALERT_COOLDOWN_SECONDS:
                        alert_triggered_for_this_person_cooldown = True
                        last_alert_time['WARNING']['Stranger'] = current_time # 更新上次报警时间

                elif res['person_id'] is not None and res['person_state'] == 1:
                    name_to_display = f"危险人物: {res['identity']}"
                    color = (0, 0, 255) # 红色：危险人物
                    event_type = 'FACE_DANGEROUS_PERSON_DETECTED'
                    has_dangerous_person_this_frame = True
                    # 检查危险人物是否在冷却期
                    if res['person_id'] not in last_alert_time['CRITICAL'] or \
                       current_time - last_alert_time['CRITICAL'][res['person_id']] > ALERT_COOLDOWN_SECONDS:
                        alert_triggered_for_this_person_cooldown = True
                        last_alert_time['CRITICAL'][res['person_id']] = current_time # 更新上次报警时间

                elif res['identity'] == 'unknown_feature_error' or res['person_id'] is None:
                    # 如果识别结果是特征提取错误或无法匹配（但已通过活体检测）
                    name_to_display = "识别错误"
                    color = (255, 0, 0) # 蓝色：错误
                    event_type = 'FACE_PROCESSING_ERROR'
                    alert_triggered_for_this_person_cooldown = True # 错误每次都记录和触发，不设冷却

                else: # 已知非危险人物
                    name_to_display = res['identity']
                    color = (0, 255, 0) # 绿色
                    # 已知合法用户的报警通常只记录日志，不触发外部高优先级报警
                    if res['person_id'] not in last_alert_time['INFO'] or \
                       current_time - last_alert_time['INFO'][res['person_id']] > ALERT_COOLDOWN_SECONDS:
                        alert_triggered_for_this_person_cooldown = True
                        last_alert_time['INFO'][res['person_id']] = current_time

                # 记录日志到数据库 (基于冷却机制判断是否触发)
                if alert_triggered_for_this_person_cooldown:
                    vision_service_worker._log_recognition_to_db({
                        'event_type': event_type,
                        'confidence': float(res.get('confidence', 0.0)),
                        'image_path': f'stream_frame_{frame_count}', # 标识帧来源
                        'camera_id': camera_id,
                        'person_id': res['person_id'],
                        'video_clip_path': log_video_clip_path,
                        'identity': res['identity']
                    })

                    # 发送外部 HTTP 报警 (仅针对 CRITICAL, WARNING 和 ERROR 级别)
                    if event_type in ['FACE_DANGEROUS_PERSON_DETECTED', 'FACE_STRANGER_DETECTED', 'FACE_PROCESSING_ERROR']:
                        alert_level_for_http = "CRITICAL" if event_type == 'FACE_DANGEROUS_PERSON_DETECTED' else \
                                               ("WARNING" if event_type == 'FACE_STRANGER_DETECTED' else "ERROR")
                        send_alert_to_main_app({
                            "alert_type": event_type,
                            "alert_level": alert_level_for_http,
                            "camera_id": camera_id,
                            "person_name": res['identity'] if res['identity'] != 'unknown_feature_error' else alert_detail_prefix,
                            "person_id": res['person_id'],
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        })


                current_frame_identities_on_screen.append(name_to_display)

                # 在人脸周围绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 添加文本标签 (姓名, 置信度, 距离)
                label = f"{name_to_display} C:{res['confidence']:.2f} D:{res['distance']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 在帧顶部添加一个整体的报警信息（类似火车站的大屏幕提示）
            overall_message = ""
            if has_dangerous_person_this_frame:
                overall_message = "【紧急警报】发现危险人员！"
            elif has_stranger_this_frame:
                overall_message = "【警告】发现陌生人！"
            elif current_frame_identities_on_screen:
                # 过滤掉“陌生人”和“识别错误”的标识，只显示已知合法用户
                known_names_for_display = [n for n in current_frame_identities_on_screen if n not in ["陌生人", "识别错误"] and not n.startswith("危险人物")]
                if known_names_for_display:
                    overall_message = f"【正常】已识别: {', '.join(set(known_names_for_display))}"
                else:
                    overall_message = "正在识别..." # 可能是检测到人脸但全是陌生人或错误
            else:
                overall_message = "未检测到人脸"

            # 在视频流顶部绘制全局消息
            cv2.putText(frame, overall_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # 白色文字，粗体


        # --- 3. 编码处理后的帧为 JPEG 格式并返回 ---
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()

        # 构造并生成响应体的一部分（MJPEG 帧）
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

    cap.release() # 循环结束时释放摄像头资源

# --- Flask 路由定义 ---

# /video_feed/<int:camera_id> 路由：提供 MJPEG 流给前端
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    try:
        # !!! 重要：这里需要替换为您的 RTMP 服务器的实际 IP 地址或域名 !!!
        # 可以从环境变量 RTMP_SERVER_IP 获取，或直接写死。
        rtmp_server_ip = os.environ.get('RTMP_SERVER_IP', '127.0.0.1')
        stream_url = f'rtmp://{rtmp_server_ip}:9090/live/{camera_id}'

        app.logger.info(f"收到摄像头 {camera_id} 的视频流请求。RTMP 流地址: {stream_url}")
        # 返回 MJPEG 流
        return Response(gen_frames(stream_url, camera_id), # 将 camera_id 传入 gen_frames
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"处理摄像头 {camera_id} 视频流失败: {str(e)}")
        return jsonify({'error': '处理视频流失败'}), 500

# /process-video 接口：用于单帧图片处理 (不处理视频URL)
@app.route('/process-video', methods=['POST'])
def process_video_task():
    app.logger.info(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 收到单帧图像处理任务 ===")

    data = request.json
    image_base64 = data.get('image_data')
    camera_id = data.get('camera_id', 0) # 从请求中获取 camera_id，默认为 0

    if not image_base64:
        return jsonify({'status': 'error', 'message': '请求必须提供 image_data。'}), 400

    frame = vision_service_worker.base64_to_cv2_image(image_base64)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Base64 图像解码失败。请检查图像格式。'}), 400

    # 进行人脸检测 (活体检测前)
    detected_faces_for_liveness = vision_service_worker.detect_faces(frame)

    # 活体检测
    is_live_person = vision_service_worker.perform_liveness_check(frame, detected_faces_for_liveness)
    if not is_live_person:
        app.logger.warning(f"单帧图像活体检测失败！疑似欺骗攻击。")
        # 记录活体欺骗事件到数据库
        vision_service_worker._log_recognition_to_db({
            'event_type': 'LIVENESS_FRAUD_DETECTED',
            'confidence': 1.0,
            'image_path': 'single_frame_liveness_test',
            'camera_id': camera_id,
            'person_id': None,
            'video_clip_path': 'N/A', # 单帧图像无视频片段路径
            'identity': 'Liveness_Fraud'
        })
        # 发送报警到主应用
        send_alert_to_main_app({
            "alert_type": "LIVENESS_FRAUD_DETECTED",
            "alert_level": "CRITICAL",
            "camera_id": camera_id,
            "person_name": "欺骗攻击",
            "person_id": None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        return jsonify({
            'status': 'alert',
            'message': '【欺骗攻击】检测到非活体！',
            'recognition_results': [] # 不返回识别结果
        }), 200


    # 活体检测通过后，进行人脸识别
    recognition_results = vision_service_worker.recognize_face_identity(frame, vision_service_worker.known_faces_data)

    alerts_triggered = [] # 记录当前请求触发的警报
    has_dangerous_person = False
    has_stranger = False
    known_persons = set()

    for res in recognition_results:
        event_type = 'UNKNOWN'
        alert_level = "INFO"
        alert_detail_prefix = ""

        if res['identity'] == 'Stranger':
            event_type = 'FACE_STRANGER_DETECTED'
            alert_level = "WARNING"
            alert_detail_prefix = "陌生人"
            has_stranger = True
        elif res['person_id'] is not None:
            if res['person_state'] == 1:
                event_type = 'FACE_DANGEROUS_PERSON_DETECTED'
                alert_level = "CRITICAL"
                alert_detail_prefix = "危险人物"
                has_dangerous_person = True
            else:
                event_type = 'FACE_KNOWN_PERSON_DETECTED'
                alert_level = "INFO"
                alert_detail_prefix = "已知人员"
                if res['identity'] != 'unknown_feature_error':
                    known_persons.add(res['identity'])
        else: # 识别错误
            event_type = 'FACE_PROCESSING_ERROR'
            alert_level = "ERROR"
            alert_detail_prefix = "识别错误"

        alert_data_for_response = {
            "alert_type": event_type,
            "alert_level": alert_level,
            "camera_id": camera_id,
            "alert_detail": f"{alert_detail_prefix}识别: {res['identity']}, 置信度: {res['confidence']:.2f}, 距离: {res['distance']:.2f}",
            "person_id": res['person_id']
        }
        alerts_triggered.append(alert_data_for_response)

        # 直接记录日志到数据库 (单帧模式下每次都记录，不设冷却)
        vision_service_worker._log_recognition_to_db({
            'event_type': event_type,
            'confidence': float(res.get('confidence', 0.0)),
            'image_path': 'base64_image_frame',
            'camera_id': camera_id,
            'person_id': res['person_id'],
            'video_clip_path': 'single_frame_data',
            'identity': res['identity']
        })

        # 发送外部 HTTP 报警 (单帧模式下，每次都发)
        if event_type in ['FACE_DANGEROUS_PERSON_DETECTED', 'FACE_STRANGER_DETECTED', 'FACE_PROCESSING_ERROR']:
            alert_level_for_http = "CRITICAL" if event_type == 'FACE_DANGEROUS_PERSON_DETECTED' else \
                                   ("WARNING" if event_type == 'FACE_STRANGER_DETECTED' else "ERROR")
            send_alert_to_main_app({
                "alert_type": event_type,
                "alert_level": alert_level_for_http,
                "camera_id": camera_id,
                "person_name": res['identity'] if res['identity'] != 'unknown_feature_error' else alert_detail_prefix,
                "person_id": res['person_id'],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })


    final_summary_message = "图像处理完成。"
    if has_dangerous_person:
        final_summary_message = "【紧急警报】发现危险人员，请立即采取措施！"
    elif has_stranger:
        final_summary_message = "【警告】发现陌生人闯入，请注意！"
    elif known_persons:
        names = "、".join(list(known_persons))
        final_summary_message = f"【正常】已识别合法用户：{names}。"
    else:
        final_summary_message = "未识别到有效人脸或处理异常。"

    return jsonify({
        'status': 'success',
        'message': final_summary_message,
        'recognition_results': recognition_results,
        'alerts_triggered': alerts_triggered
    })


# /extract-embedding 路由：用于提取人脸特征向量
@app.route('/extract-embedding', methods=['POST'])
def extract_embedding_from_image():
    data = request.json
    image_base64 = data.get('image_data')

    if not image_base64:
        return jsonify({'status': 'error', 'message': '未提供图像数据。'}), 400

    frame = vision_service_worker.base64_to_cv2_image(image_base64)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Base64 图像解码失败。请检查图像格式。'}), 400

    detected_faces = vision_service_worker.detect_faces(frame)
    if not detected_faces:
        return jsonify({'status': 'error', 'message': '图像中未检测到人脸。'}), 400

    # 假设提取第一个检测到的人脸的特征向量
    x1, y1, x2, y2 = detected_faces[0]['box_coords']
    h, w = frame.shape[:2]
    cropped_face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    if cropped_face.size == 0:
        return jsonify({'status': 'error', 'message': '裁剪的人脸区域为空。无法提取特征。'}), 500

    face_embedding = vision_service_worker.extract_face_features(cropped_face)

    if face_embedding.size == 0:
        return jsonify({'status': 'error', 'message': '无法从图像中提取人脸特征。'}), 500

    return jsonify({
        'status': 'success',
        'message': '人脸特征提取成功。',
        'face_embedding': face_embedding.tolist(), # 将 NumPy 数组转换为列表以便 JSON 序列化
        'box_coords': detected_faces[0]['box_coords'] # 返回检测到的边界框
    })

# /register-face 路由：用于注册新的人脸信息
@app.route('/register-face', methods=['POST'])
def register_face():
    app.logger.info(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 收到新的人脸注册请求 ===")

    data = request.json
    image_base64 = data.get('image_data')
    person_name = data.get('person_name')
    person_state_input = data.get('state', 0) # 允许从请求中传入 state，默认为 0

    if not image_base64:
        return jsonify({'status': 'error', 'message': '未提供图像数据 (image_data)。'}), 400
    if not person_name or not isinstance(person_name, str) or len(person_name.strip()) == 0:
        return jsonify({'status': 'error', 'message': '未提供有效的人员姓名 (person_name)。'}), 400
    try:
        person_state = int(person_state_input)
        if person_state not in [0, 1]:
            raise ValueError("State must be 0 or 1.")
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': '人员状态 (state) 必须是 0 或 1。'}), 400

    frame = vision_service_worker.base64_to_cv2_image(image_base64)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Base64 图像解码失败。请检查图像格式。'}), 400

    detected_faces = vision_service_worker.detect_faces(frame)
    if not detected_faces:
        return jsonify({'status': 'error', 'message': '图像中未检测到人脸。请确保图像清晰并包含完整人脸。'}), 400
    
    # 假设只处理检测到的第一张人脸进行注册
    x1, y1, x2, y2 = detected_faces[0]['box_coords']
    h, w = frame.shape[:2]
    cropped_face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] 

    if cropped_face.size == 0:
        return jsonify({'status': 'error', 'message': '裁剪的人脸区域为空。无法提取特征。'}), 500

    face_embedding = vision_service_worker.extract_face_features(cropped_face)
    if face_embedding.size == 0:
        return jsonify({'status': 'error', 'message': '无法从图像中提取人脸特征。'}), 500

    face_embedding_list = face_embedding.tolist()

    safe_person_name = person_name.strip().replace(" ", "_").lower()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    # 图像路径（在实际部署中，可能需要将裁剪的人脸图像保存到文件系统或云存储）
    image_path_for_db = f'/media/faces/{safe_person_name}_{timestamp}.jpg' 
    
    new_person_id = vision_service_worker._save_new_person_to_db(person_name, person_state, image_path_for_db, face_embedding_list)

    if new_person_id is not None:
        # 成功注册后，刷新 Worker 中的已知人脸数据缓存
        vision_service_worker.known_faces_data = vision_service_worker._load_known_faces()
        
        return jsonify({
            'status': 'success',
            'message': f'人脸信息成功注册，用户: {person_name}, ID: {new_person_id}, State: {person_state}',
            'person_id': new_person_id,
            'face_embedding': face_embedding_list,
            'face_image_path': image_path_for_db
        }), 201 # HTTP 201 Created 状态码表示资源已成功创建
    else:
        return jsonify({'status': 'error', 'message': '人脸信息注册失败。请检查后端日志。'}), 500


# --- Flask 应用入口点 ---
if __name__ == '__main__':
    app.logger.info(">>> 人脸识别视频流 Worker 正在启动，监听 http://0.0.0.0:5000")
    # 在运行 Worker 之前，请确保设置以下环境变量：
    # 1. RTMP_SERVER_IP: 您的 RTMP 流媒体服务器的 IP 地址 (例如 "192.168.1.100" 或 "127.0.0.1")
    #    Linux/macOS: export RTMP_SERVER_IP="127.0.0.1"
    #    Windows CMD: set RTMP_SERVER_IP="127.0.0.1"
    #    Windows PowerShell: $env:RTMP_SERVER_IP="127.0.0.1"
    # 2. ALERT_RECEIVER_URL: 您主应用 (如 mock_main_app.py) 接收报警的接口地址
    #    Linux/macOS: export ALERT_RECEIVER_URL="http://127.0.0.1:8000/api/alerts/receive/"
    #    Windows CMD: set ALERT_RECEIVER_URL="http://127.0.0.1:8000/api/alerts/receive/"
    #    Windows PowerShell: $env:ALERT_RECEIVER_URL="http://127.0.0.1:8000/api/alerts/receive/"
    # 3. DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT: 您的 PostgreSQL 数据库连接信息

    # For production deployment, use a production-ready WSGI server like Gunicorn or uWSGI.
    # Debug mode should be disabled in production due to security risks and performance overhead.
    app.run(host='0.0.0.0', port=5000, debug=True) # 开发/测试时可启用调试模式