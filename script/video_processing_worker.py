"""

人脸识别微服务（recognition-api）的 Flask 应用，负责处理视频流和图像数据，执行人脸检测、特征提取和比对识别。

- 加载预训练的深度学习模型，用于人脸检测和特征提取。
- 连接 PostgreSQL 数据库，加载已知人脸的特征数据。
- 提供 API 接口，用于接收视频 URL 或 Base64 图像，执行人脸识别任务。
- 将所有成功的识别事件记录到数据库的 'recognition_log' 表中。
- 提供 API 接口，用于将新个体的人脸信息（图像和姓名）注册到 'person' 数据库表中。
- 提供辅助 API，用于从图像中单独提取人脸特征向量。
"""

import os
import requests
from flask import Flask, request, jsonify
from tqdm import tqdm
from tqdm import tqdm
import datetime

import cv2
import numpy as np
import base64
from scipy.spatial.distance import euclidean

# PostgreSQL 数据库连接库
import psycopg2
# 用于处理 JSON 序列化/反序列化的库（例如，face_embedding 字段存储为 TEXT 类型）
import json

# 初始化 Flask 应用实例
app = Flask(__name__)

# 定义本地视频文件的临时下载目录
DOWNLOAD_DIR = "received_videos"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# --- Worker 核心配置 ---
# 主 Django 应用的 Base URL。
# 此处为占位符，用于未来可能的 Worker 回调或数据交换机制。
# 在生产环境中应通过环境变量进行配置。
DJANGO_MAIN_APP_BASE_URL = os.environ.get('DJANGO_MAIN_APP_BASE_URL', 'http://your_django_app_host:8000')

class VisionServiceWorker:
    """
    单例类，管理人脸识别任务的计算机视觉模型和数据库交互。
    确保模型只加载一次。
    """
    _instance = None # 用于存储类的唯一实例

    def __new__(cls, *args, **kwargs):
        """实现单例模式的 __new__ 方法。"""
        if not cls._instance:
            cls._instance = super(VisionServiceWorker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """
        初始化模型、数据库连接参数并加载已知人脸数据。
        此方法被设计为在单例实例首次创建时只调用一次。
        """
        # 使用 _initialized 标志防止在后续调用 __init__ 时重复初始化
        if not hasattr(self, '_initialized'):
            self._initialized = True

            # --- 模型文件路径配置 ---
            # 动态计算模型目录：从当前脚本目录 ('script/') 向上导航一层 (recognition-api/)，然后进入 'dnn_models/'
            current_script_dir = os.path.abspath(os.path.dirname(__file__))
            self.MODEL_DIR = os.path.join(current_script_dir, '..', 'dnn_models')
            
            # 人脸检测模型文件路径 (Caffe 模型)
            self.FACE_DETECTOR_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector.pbtxt')
            self.FACE_DETECTOR_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector_uint8.pb')
            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.7 # 识别人脸的最低置信度阈值

            # 人脸识别模型文件路径 (Torch 模型)
            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, 'nn4.small2.v1.t7') 
            self.FACE_RECOGNITION_THRESHOLD = 0.8  # 欧氏距离阈值，用于判断是否为同一个人（距离越小越相似）

            # 模型实例，初始化为 None，在 _load_all_models() 中加载
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None

            # --- 数据库连接参数 ---
            # 在生产环境中，强烈建议通过环境变量配置这些敏感凭证。
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
            print(f"错误: 无法连接到数据库。详情: {e}")
            return None

    def _load_all_models(self):
        """
        加载人脸检测和人脸识别的深度学习模型。
        如果模型无法加载，则抛出 RuntimeError，阻止 Worker 启动。
        """
        print("Worker: 正在加载人脸相关深度学习模型...")
        try:
            # 在尝试加载之前，检查模型文件是否存在
            if not os.path.exists(self.FACE_DETECTOR_PROTOTXT_PATH):
                raise FileNotFoundError(f"人脸检测模型 prototxt 文件未找到: {self.FACE_DETECTOR_PROTOTXT_PATH}")
            if not os.path.exists(self.FACE_DETECTOR_WEIGHTS_PATH):
                raise FileNotFoundError(f"人脸检测模型权重文件未找到: {self.FACE_DETECTOR_WEIGHTS_PATH}")
            if not os.path.exists(self.FACE_RECOGNITION_MODEL_PATH):
                raise FileNotFoundError(f"人脸识别模型文件未找到: {self.FACE_RECOGNITION_MODEL_PATH}")

            # 加载用于人脸检测的 Caffe 模型
            self.FACE_DETECTOR_NET = cv2.dnn.readNet(
                self.FACE_DETECTOR_PROTOTXT_PATH,
                self.FACE_DETECTOR_WEIGHTS_PATH
            )
            print("Worker: 人脸检测模型加载完成。")

            # 加载用于人脸特征提取的 Torch 模型
            self.FACE_RECOGNITION_NET = cv2.dnn.readNetFromTorch(
                self.FACE_RECOGNITION_MODEL_PATH
            )
            print("Worker: 人脸特征提取模型加载完成。")

        except Exception as e:
            print(f"错误: Worker: 无法加载一个或多个人脸模型。详情: {e}")
            print(f"请确保模型文件位于: '{os.path.abspath(self.MODEL_DIR)}' 目录下。")
            # 抛出 RuntimeError 以阻止 Worker 启动，因为缺少模型无法正常工作
            raise RuntimeError("人脸识别模型加载失败，请检查路径和文件。")

    def _load_known_faces(self) -> list:
        """
        从 PostgreSQL 的 'person' 表加载已知人脸数据（人员 ID、姓名、人脸特征向量）。
        将存储为 JSON 字符串的特征向量反序列化为 NumPy 数组。
        """
        print("Worker: 正在从数据库加载已知人脸数据...")
        known_faces_parsed = []
        conn = None
        try:
            conn = self._get_db_connection()
            if conn:
                cur = conn.cursor()
                # 从 'person' 表中选择 ID、姓名、图片路径和人脸特征向量
                cur.execute("SELECT id, name, face_image_path, face_embedding FROM person;")
                rows = cur.fetchall()
                cur.close()

                for row in rows:
                    person_id, name, face_image_path, face_embedding_str = row
                    if face_embedding_str:
                        try:
                            # 将 JSON 字符串形式的特征向量反序列化为 Python 列表，然后转换为 NumPy 数组
                            embedding_list = json.loads(face_embedding_str)
                            face_embedding = np.array(embedding_list, dtype=np.float32)
                            known_faces_parsed.append({
                                'id': person_id,
                                'name': name,
                                'face_image_path': face_image_path,
                                'face_embedding': face_embedding
                            })
                        except json.JSONDecodeError as jde:
                            print(f"警告: 无法解析 {name} (ID: {person_id}) 的 face_embedding。详情: {jde}")
                    else:
                        print(f"警告: 已知人脸数据 {name} (ID: {person_id}) 缺少 face_embedding。")

                print(f"Worker: 已从数据库成功加载 {len(known_faces_parsed)} 条已知人脸数据。")
        except Exception as e:
            print(f"错误: Worker: 无法从数据库检索已知人脸数据。详情: {e}")
            print("Worker: 将使用空的人脸数据列表，人脸识别功能将受到限制。")
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
            print(f"错误: 解码 Base64 图片失败: {e}")
            return None

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        在图像帧中检测人脸。
        返回一个字典列表，每个字典包含人脸的边界框坐标和置信度。
        """
        if self.FACE_DETECTOR_NET is None:
            print("警告: 人脸检测模型未加载。无法检测人脸。")
            return []

        (h, w) = frame.shape[:2] # 获取帧的尺寸
        # 从图像创建用于 DNN 输入的 blob（调整大小为 300x300，进行均值减法）
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.FACE_DETECTOR_NET.setInput(blob)
        detections = self.FACE_DETECTOR_NET.forward() # 执行检测

        detected_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2] # 获取检测置信度
            if confidence > self.FACE_DETECTOR_CONFIDENCE_THRESHOLD:
                # 将边界框坐标缩放回原始帧尺寸
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int") # 转换为整数坐标（可能是 numpy.int64 类型）

                detected_faces.append({
                    # 确保坐标是标准的 Python int 类型，以便 JSON 序列化
                    'box_coords': [int(startX), int(startY), int(endX), int(endY)],
                    # 确保置信度是标准的 Python float 类型，以便 JSON 序列化
                    'confidence': float(confidence)
                })
        return detected_faces

    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        从裁剪出的人脸图像中提取 128 维人脸特征向量。
        """
        if self.FACE_RECOGNITION_NET is None:
            print("警告: 人脸特征提取模型未加载。无法提取特征。")
            return np.array([])

        # 对人脸图像进行预处理，以符合识别模型的输入要求（调整大小到 96x96，归一化像素）
        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (96, 96)), 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.FACE_RECOGNITION_NET.setInput(face_blob)
        features = self.FACE_RECOGNITION_NET.forward()
        return features.flatten() # 返回展平的 128 维向量

    def recognize_face_identity(self, frame: np.ndarray, known_faces_data: list) -> list:
        """
        在图像帧中检测人脸，提取其特征，并与已知人脸进行比对。
        返回每个检测到人脸的识别结果列表。
        """
        results = []
        # 调用 detect_faces 确保返回的 box_coords 已经是标准的 Python int 列表
        detected_faces = self.detect_faces(frame) 

        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            
            # 确保裁剪区域有效且不超出帧边界
            h, w = frame.shape[:2]
            startY, endY = max(0, y1), min(h, y2)
            startX, endX = max(0, x1), min(w, x2)

            cropped_face = frame[startY:endY, startX:endX]

            if cropped_face.size == 0:
                print("警告: 裁剪到的人脸区域为空。跳过特征提取。")
                results.append({
                    'box_coords': face_info['box_coords'], # 已经是 int 列表
                    'confidence': float(face_info['confidence']),
                    'identity': 'unknown_feature_error', # 指示特征提取错误
                    'distance': -1.0, # 用 -1.0 浮点数表示错误
                    'person_id': None
                })
                continue

            face_embedding = self.extract_face_features(cropped_face)

            if face_embedding.size == 0:
                print("警告: 无法从人脸区域提取特征。")
                results.append({
                    'box_coords': face_info['box_coords'], # 已经是 int 列表
                    'confidence': float(face_info['confidence']),
                    'identity': 'unknown_feature_error', # 指示特征提取错误
                    'distance': -1.0, # 用 -1.0 浮点数表示错误
                    'person_id': None
                })
                continue

            min_distance = float('inf') # 初始化最小距离为无穷大
            identity = 'Stranger'      # 默认身份为“陌生人”
            person_id = None           # 默认人员 ID 为 None

            # 遍历所有已知人脸的特征数据进行比对
            for known_face in known_faces_data:
                known_embedding = known_face['face_embedding'] 

                # 检查特征向量维度是否匹配（通常不应该发生）
                if known_embedding.shape != face_embedding.shape:
                    print(f"警告: 特征向量维度不匹配。已知维度: {known_embedding.shape}, 新维度: {face_embedding.shape}")
                    continue

                # 计算当前人脸特征向量与已知人脸特征向量之间的欧氏距离
                distance = euclidean(face_embedding, known_embedding) 
                
                # 如果发现更小的距离，则更新最佳匹配
                if distance < min_distance:
                    min_distance = distance
                    identity = known_face['name'] 
                    person_id = known_face['id'] 

            # 根据识别阈值判断最终身份
            if min_distance < self.FACE_RECOGNITION_THRESHOLD:
                # 如果最小距离小于阈值，则保留当前最佳匹配的身份和 ID
                pass 
            else:
                # 如果最佳距离大于或等于阈值，则分类为“陌生人”
                identity = 'Stranger' 
                person_id = None 

            results.append({
                'box_coords': face_info['box_coords'],       # Python int 列表
                'confidence': float(face_info['confidence']), # Python float
                'identity': identity,                         # 字符串
                'distance': float(min_distance),              # Python float
                'person_id': person_id                        # Python int 或 None
            })
        
        return results

    def _log_recognition_to_db(self, log_data: dict):
        """
        将人脸识别事件记录到 PostgreSQL 数据库的 'recognition_log' 表中。
        """
        conn = None
        try:
            conn = self._get_db_connection()
            if conn:
                cur = conn.cursor()
                insert_query = """
                INSERT INTO recognition_log (time, confidence, image_path, camera_id, person_id, video_clip)
                VALUES (%s, %s, %s, %s, %s, %s);
                """
                # 'time' 字段是 TIMESTAMP WITH TIME ZONE 类型，使用带时区的 datetime 对象
                cur.execute(insert_query, (
                    datetime.datetime.now(datetime.timezone.utc), # 使用 UTC 时区以保证时间一致性
                    log_data.get('confidence'),
                    log_data.get('image_path', ''), # 图像标识符（例如，帧号，'base64_image_frame'）
                    log_data.get('camera_id'),
                    log_data.get('person_id'),
                    log_data.get('video_clip', '') # 视频片段标识符（例如，文件名，URL）
                ))
                conn.commit() # 提交事务以保存更改
                cur.close()
                print(f"Worker: 识别日志已成功写入数据库: {log_data.get('identity', 'Stranger')} (人员 ID: {log_data.get('person_id')})")
        except Exception as e:
            print(f"错误: 写入识别日志到数据库失败。详情: {e}")
            if conn: # 确保在发生错误时回滚事务
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def _save_new_person_to_db(self, person_name: str, face_image_path: str, face_embedding_list: list) -> int | None:
        """
        将新录入的人脸信息（姓名、图像路径、特征向量）保存到 'person' 表中。
        ID 由数据库自动生成。
        返回新生成的 person_id 或 None (如果保存失败)。
        """
        conn = None
        new_person_id = None
        try:
            conn = self._get_db_connection()
            if conn:
                cur = conn.cursor()
                # 将特征向量列表转换为 JSON 字符串，以便存储到 TEXT 类型列
                face_embedding_json_str = json.dumps(face_embedding_list)

                # INSERT 语句明确不包含 'id' 列，因为它是由数据库自动生成的 (GENERATED BY DEFAULT AS IDENTITY)。
                # 使用 RETURNING id 子句来获取新插入行的自动生成 ID。
                insert_query = """
                INSERT INTO person (name, face_image_path, face_embedding)
                VALUES (%s, %s, %s) RETURNING id;
                """
                cur.execute(insert_query, (
                    person_name,
                    face_image_path,
                    face_embedding_json_str
                ))
                new_person_id = cur.fetchone()[0] # 获取自动生成的 ID
                conn.commit() # 提交事务
                cur.close()
                print(f"Worker: 成功将新人员 '{person_name}' (自动生成 ID: {new_person_id}) 保存到数据库。")
                return new_person_id
        except Exception as e:
            print(f"错误: 将新人员信息保存到数据库失败。详情: {e}")
            print("请检查数据库连接、表结构或是否存在违反唯一约束的情况。")
            if conn:
                conn.rollback() # 发生错误时回滚事务
        finally:
            if conn:
                conn.close()
        return None


# 创建 VisionServiceWorker 单例实例。
# 模型和已知人脸数据将在 Worker 启动时进行加载。
vision_service_worker = VisionServiceWorker()


# 辅助函数：从 URL 下载视频文件到指定目录。
def download_video(url: str, directory: str) -> str | None:
    """从 URL 下载视频文件到指定的本地目录。"""
    # tqdm 在此函数内局部导入，以避免不必要的全局依赖。
    from tqdm import tqdm
    try:
        # 从 URL 中提取文件名作为本地保存的文件名
        local_filename = os.path.join(directory, url.split('/')[-1])
        print(f"正在下载视频: {url}")
        print(f"保存至: {local_filename}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status() # 对非 2xx 的 HTTP 状态码抛出 HTTPError 异常
            total_size = int(r.headers.get('content-length', 0)) # 从响应头获取文件总大小
            
            # 使用 tqdm 显示下载进度条
            with open(local_filename, 'wb') as f, tqdm(
                    desc=local_filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192): # 以 8KB 为块迭代响应内容
                    size = f.write(chunk)
                    bar.update(size) # 更新进度条
        print(f"视频成功下载: '{local_filename}'。")
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"错误: 视频下载失败，URL: {url}。详情: {e}")
        return None

# --- Flask 路由定义 ---

@app.route('/process-video', methods=['POST'])
def process_video_task():
    """
    API 接口，用于处理视频 URL 或 Base64 图像，执行人脸识别。
    执行人脸检测、识别，并将事件记录到数据库。
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] === 收到新的视频/图像处理任务 ===")

    data = request.json
    video_url = data.get('video_url')
    # camera_id 在数据库中是 INTEGER 类型，确保从请求中获取的是整数
    camera_id = data.get('camera_id', 0) 

    # 用于日志记录的视频片段标识符（来自 URL 文件名或 'single_frame_image_data'）
    log_video_clip_identifier = video_url.split('/')[-1] if video_url else 'single_frame_image_data'

    if not video_url:
        # 如果未提供 video_url，则尝试处理 Base64 编码的图像数据
        image_base64 = data.get('image_data')
        if not image_base64:
            return jsonify({'status': 'error', 'message': '请求必须提供 video_url 或 image_data。'}), 400
        
        # 将 Base64 图像解码为 OpenCV 图像帧
        frame = vision_service_worker.base64_to_cv2_image(image_base64)
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Base64 图像解码失败。请检查图像格式。'}), 400
        
        # 对单帧图像执行人脸识别
        recognition_results = vision_service_worker.recognize_face_identity(frame, vision_service_worker.known_faces_data)
        
        alerts_triggered = []
        for res in recognition_results:
            # 如果识别到已知人员（非陌生人且无特征提取错误）且 person_id 存在
            if res['identity'] != 'Stranger' and res['identity'] != 'unknown_feature_error' and res['person_id'] is not None:
                alert_data = {
                    "alert_type": "FACE_RECOGNIZED",
                    "alert_level": "INFO",
                    "camera_id": camera_id,
                    "alert_detail": f"用户识别: {res['identity']}, 置信度: {res['confidence']:.2f}",
                    "person_id": res['person_id'] 
                }
                alerts_triggered.append(alert_data)
                
                # 记录识别事件到数据库
                vision_service_worker._log_recognition_to_db({
                    'confidence': res['confidence'],
                    'image_path': 'base64_image_frame', # 图像来源标识符
                    'camera_id': camera_id,
                    'person_id': res['person_id'],
                    'video_clip': log_video_clip_identifier,
                    'identity': res['identity'] # 用于控制台日志消息
                })

        return jsonify({
            'status': 'success',
            'message': '单帧图像处理成功。',
            'recognition_results': recognition_results,
            'alerts_triggered': alerts_triggered
        })
    else:
        # 如果提供了 video_url，则下载并处理视频文件
        saved_path = download_video(video_url, DOWNLOAD_DIR)

        if saved_path:
            cap = cv2.VideoCapture(saved_path)
            if not cap.isOpened():
                return jsonify({'status': 'error', 'message': f'无法打开下载的视频文件: {saved_path}'}), 500

            all_recognition_results = []
            alerts_from_video = []
            processed_frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break # 视频流结束

                # 每处理 5 帧，执行一次人脸识别以降低处理负载
                if processed_frame_count % 5 == 0: 
                    results_per_frame = vision_service_worker.recognize_face_identity(frame, vision_service_worker.known_faces_data)
                    all_recognition_results.extend(results_per_frame)

                    for res in results_per_frame:
                        # 如果识别到已知人员（非陌生人且无特征提取错误）且 person_id 存在
                        if res['identity'] != 'Stranger' and res['identity'] != 'unknown_feature_error' and res['person_id'] is not None:
                            alert_data = {
                                "alert_type": "FACE_RECOGNIZED_VIDEO",
                                "alert_level": "MEDIUM",
                                "camera_id": camera_id,
                                "alert_detail": f"视频识别用户: {res['identity']} (置信度: {res['confidence']:.2f})",
                                "person_id": res['person_id'] 
                            }
                            alerts_from_video.append(alert_data)

                            # 记录识别事件到数据库
                            vision_service_worker._log_recognition_to_db({
                                'confidence': res['confidence'],
                                'image_path': f'frame_{processed_frame_count}', # 按帧号标识图像来源
                                'camera_id': camera_id,
                                'person_id': res['person_id'],
                                'video_clip': log_video_clip_identifier,
                                'identity': res['identity'] # 用于控制台日志消息
                            })

                processed_frame_count += 1

            cap.release()
            
            # 可选：处理完成后删除下载的视频文件（如果不需要保留，请取消注释）
            # os.remove(saved_path) 

            return jsonify({
                'status': 'success',
                'message': '视频处理成功。',
                'processed_video_path': saved_path, # 返回下载路径以供参考
                'recognition_results': all_recognition_results,
                'alerts_triggered': alerts_from_video
            })
        else:
            return jsonify({'status': 'error', 'message': '视频下载失败。'}), 500


@app.route('/extract-embedding', methods=['POST'])
def extract_embedding_from_image():
    """
    API 接口，用于从 Base64 编码图像中提取 128 维人脸特征向量。
    主要用作辅助工具或调试目的。
    """
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

    # 假设提取第一个检测到的人脸的特征向量。
    # 对于多张人脸，可能需要更复杂的逻辑（例如，选择特定人脸）。
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

@app.route('/register-face', methods=['POST'])
def register_face():
    """
    API 接口，用于将新的人脸信息注册到系统中。
    接收来自前端的 Base64 图像和人员姓名，
    提取人脸特征并将其保存到 'person' 数据库表中。
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] === 收到新的人脸注册请求 ===")

    data = request.json
    image_base64 = data.get('image_data')
    person_name = data.get('person_name')

    # 输入验证
    if not image_base64:
        return jsonify({'status': 'error', 'message': '未提供图像数据 (image_data)。'}), 400
    if not person_name or not isinstance(person_name, str) or len(person_name.strip()) == 0:
        return jsonify({'status': 'error', 'message': '未提供有效的人员姓名 (person_name)。'}), 400

    # 1. 解码 Base64 图像为 OpenCV 帧
    frame = vision_service_worker.base64_to_cv2_image(image_base64)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Base64 图像解码失败。请检查图像格式。'}), 400

    # 2. 在图像中检测人脸
    detected_faces = vision_service_worker.detect_faces(frame)
    if not detected_faces:
        return jsonify({'status': 'error', 'message': '图像中未检测到人脸。请确保图像清晰并包含完整人脸。'}), 400
    
    # 假设只处理检测到的第一张人脸进行注册。
    # 在生产环境中，对于多张人脸可能需要特定的逻辑。
    x1, y1, x2, y2 = detected_faces[0]['box_coords']
    h, w = frame.shape[:2]
    cropped_face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] 

    if cropped_face.size == 0:
        return jsonify({'status': 'error', 'message': '裁剪的人脸区域为空。无法提取特征。'}), 500

    # 3. 提取人脸特征向量
    face_embedding = vision_service_worker.extract_face_features(cropped_face)
    if face_embedding.size == 0:
        return jsonify({'status': 'error', 'message': '无法从图像中提取人脸特征。'}), 500

    # 将 NumPy 数组转换为 Python 列表以便 JSON 序列化和数据库存储
    face_embedding_list = face_embedding.tolist()

    # 4. 生成人脸图像的逻辑路径（用于数据库存储）。
    # 在完整的系统中，实际图像可能会保存到媒体文件夹或云存储中。
    safe_person_name = person_name.strip().replace(" ", "_").lower() # 清理姓名用于文件名
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f") # 使用 UTC 时间戳确保唯一性
    image_path_for_db = f'/media/faces/{safe_person_name}_{timestamp}.jpg' 
    
    # 示例：将裁剪后的人脸图像保存到本地（需要创建 'media/faces' 目录，相对于项目根目录）
    # try:
    #     save_dir = os.path.join(os.path.dirname(__file__), '..', 'media', 'faces')
    #     os.makedirs(save_dir, exist_ok=True)
    #     cv2.imwrite(os.path.join(save_dir, f"{safe_person_name}_{timestamp}.jpg"), cropped_face)
    # except Exception as e:
    #     print(f"警告: 无法将人脸图像保存到本地存储。详情: {e}")

    # 5. 将人员人脸信息保存到数据库
    new_person_id = vision_service_worker._save_new_person_to_db(person_name, image_path_for_db, face_embedding_list)

    if new_person_id is not None:
        # 成功注册后，刷新 Worker 中的已知人脸数据缓存，
        # 以确保新添加的人员能够立即被正在进行的处理任务识别。
        # 注意：这在处理高吞吐量的生产环境中可能会有性能开销，可能需要更智能的缓存刷新机制。
        vision_service_worker.known_faces_data = vision_service_worker._load_known_faces()
        
        return jsonify({
            'status': 'success',
            'message': f'人脸信息成功注册，用户: {person_name}, ID: {new_person_id}',
            'person_id': new_person_id,
            'face_embedding': face_embedding_list, # 可选：返回特征向量，供前端调试/参考
            'face_image_path': image_path_for_db
        }), 201 # HTTP 201 Created 状态码表示资源已成功创建
    else:
        return jsonify({'status': 'error', 'message': '人脸信息注册失败。请检查后端日志。'}), 500


# Flask 应用程序的入口点
if __name__ == '__main__':
    print(">>> 视频处理 Worker 正在启动，监听 http://0.0.0.0:5000")
    # 对于生产部署，请使用生产级的 WSGI 服务器，例如 Gunicorn 或 uWSGI。
    # 调试模式在生产环境中应禁用，因为它会带来安全风险和性能开销。
    # app.run(host='0.0.0.0', port=5000, debug=False) 
    app.run(host='0.0.0.0', port=5000, debug=True) # 开发/测试时可启用调试模式