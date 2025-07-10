import cv2
import sys
import os
import numpy as np
import base64
import json # 新增：用于将 NumPy 数组转换为 JSON 字符串
import psycopg2 # 新增：用于数据库操作
import datetime # 新增：用于可能的时间戳字段，虽然person表没有，但这是一个好习惯

# 将 script/ 目录添加到 Python 路径，以便导入 video_processing_worker 中的 VisionServiceWorker
# 假设 enroll_face.py 与 script 目录同级
sys.path.append(os.path.join(os.path.dirname(__file__), 'script'))
from video_processing_worker import VisionServiceWorker # 导入 VisionServiceWorker 类

# --- 配置 ---
WEBCAM_ID = 0 # 使用默认摄像头

# --- 数据库连接配置 (与 video_processing_worker.py 保持一致) ---
# 建议通过环境变量设置，这里为了演示直接使用硬编码（但生产环境应避免）
DB_HOST = os.environ.get('DB_HOST', 'your_db_host') # <<< 替换为你的数据库主机
DB_NAME = os.environ.get('DB_NAME', 'your_db_name') # <<< 替换为你的数据库名称
DB_USER = os.environ.get('DB_USER', 'your_db_user') # <<< 替换为你的数据库用户
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_db_password') # <<< 替换为你的数据库密码
DB_PORT = os.environ.get('DB_PORT', '5432') # PostgreSQL 默认端口，如果不同请替换

print(">>> 人脸特征录入工具启动 <<<")
print("请将您的脸部对准摄像头，然后按下 's' 键捕捉，或按 'q' 键退出。")

# 实例化 VisionServiceWorker
try:
    enroll_worker = VisionServiceWorker()
except RuntimeError as e:
    print(f"初始化人脸识别服务失败: {e}")
    sys.exit(1)


def get_db_connection():
    """建立并返回一个数据库连接."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"错误: 无法连接到数据库。详情: {e}")
        return None

# 修改此函数以适应数据库自动生成 ID
def save_person_to_db(person_name, face_image_path, face_embedding_list):
    """将人脸信息保存到 person 表中，ID由数据库自动生成。"""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # 确保 face_embedding 是 JSON 字符串
            face_embedding_json_str = json.dumps(face_embedding_list)

            # 修改 INSERT 语句：不再包含 id 列
            insert_query = """
            INSERT INTO person (name, face_image_path, face_embedding)
            VALUES (%s, %s, %s) RETURNING id; -- RETURNING id 可以获取自动生成的ID
            """
            cur.execute(insert_query, (
                person_name,
                face_image_path,
                face_embedding_json_str
            ))
            new_person_id = cur.fetchone()[0] # 获取自动生成的ID
            conn.commit()
            cur.close()
            print(f"\n成功将 {person_name} (自动生成ID: {new_person_id}) 的人脸信息保存到数据库。")
            return True
    except Exception as e:
        print(f"\n错误: 无法将人脸信息保存到数据库。详情: {e}")
        print("请检查数据库连接配置和表结构。")
        return False
    finally:
        if conn:
            conn.close()

cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print("错误: 无法打开摄像头。请确保摄像头连接正常，没有被其他程序占用。")
    sys.exit(1)

face_captured_success = False
person_name = input("请输入您要录入的姓名 (例如: 张三): ")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧。")
        break

    display_frame = frame.copy() # 复制帧用于显示，避免在原始帧上绘制
    cv2.putText(display_frame, "Press 's' to capture, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    detected_faces = enroll_worker.detect_faces(frame) # 使用 enroll_worker 实例
    if detected_faces:
        # 只处理检测到的第一张脸
        (x1, y1, x2, y2) = detected_faces[0]['box_coords']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box for detection

    cv2.imshow("Face Enrollment - Press 's' to Capture", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if detected_faces:
            (x1, y1, x2, y2) = detected_faces[0]['box_coords']
            cropped_face = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            if cropped_face.size > 0:
                print("正在提取人脸特征，请稍候...")
                face_embedding = enroll_worker.extract_face_features(cropped_face)

                if face_embedding.size > 0:
                    # 将 NumPy 数组转换为 Python 列表，以便 JSON 序列化和存储到数据库
                    face_embedding_list = face_embedding.tolist()
                    
                    # 假设 face_image_path 可以是一个占位符或实际的存储路径
                    image_path_for_db = f'/media/faces/{person_name.lower().replace(" ", "_")}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'

                    # 调用 save_person_to_db，不再传递 person_id
                    if save_person_to_db(person_name, image_path_for_db, face_embedding_list):
                        face_captured_success = True
                        break # 捕获成功并保存到数据库后退出循环
                    else:
                        print("错误: 保存到数据库失败。")
                else:
                    print("错误: 无法提取人脸特征。请确保脸部清晰且光线充足。")
            else:
                print("错误: 未检测到有效人脸，请将脸部对准摄像头，确保蓝色框出现。")
        else:
            print("未检测到人脸，请将脸部对准摄像头。")
    elif key == ord('q'):
        print("用户退出录入。")
        break

cap.release()
cv2.destroyAllWindows()

if face_captured_success:
    print("人脸录入完成，信息已存储到数据库。")
else:
    print("人脸录入未完成。")