"""
enroll_face.py

这是一个辅助命令行工具，用于通过本地摄像头捕捉人脸图像，
提取人脸特征向量，并将新用户的人脸信息直接注册到 PostgreSQL 数据库的 'person' 表中。
ID 由数据库自动生成。用户可以手动设置 state 字段（0为非危险人员，1为危险人员）。
"""

import cv2
import sys
import os
import numpy as np
import base64
import json
import psycopg2
import datetime

# 将 script/ 目录添加到 Python 路径，以便导入 video_processing_worker 中的 VisionServiceWorker。
sys.path.append(os.path.join(os.path.dirname(__file__), 'script'))
from video_processing_worker import VisionServiceWorker

# --- 配置 ---
WEBCAM_ID = 0

# --- 数据库连接配置 ---
DB_HOST = os.environ.get('DB_HOST', 'your_db_host') # <<< 替换为你的数据库主机
DB_NAME = os.environ.get('DB_NAME', 'your_db_name') # <<< 替换为你的数据库名称
DB_USER = os.environ.get('DB_USER', 'your_db_user') # <<< 替换为你的数据库用户
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_db_password') # <<< 替换为你的数据库密码
DB_PORT = os.environ.get('DB_PORT', '5432')

print(">>> 人脸特征录入工具启动 <<<")
print("请将您的脸部对准摄像头，然后按下 's' 键捕捉，或按 'q' 键退出。")

try:
    enroll_worker = VisionServiceWorker()
except RuntimeError as e:
    print(f"初始化人脸识别服务失败: {e}")
    sys.exit(1)


def get_db_connection():
    """建立并返回一个数据库连接。"""
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

# 修改 save_person_to_db 函数，使其接受 state 参数
def save_person_to_db(person_name: str, state: int, face_image_path: str, face_embedding_list: list) -> int | None:
    """
    将新录入的人脸信息保存到 'person' 表中。
    ID 由数据库自动生成。state 由调用者传入。
    返回新生成的 person_id 或 None (如果保存失败)。
    """
    conn = None
    new_person_id = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            face_embedding_json_str = json.dumps(face_embedding_list)

            insert_query = """
            INSERT INTO person (name, state, face_image_path, face_embedding)
            VALUES (%s, %s, %s, %s) RETURNING id;
            """
            # 将传入的 state 参数用于 SQL 插入
            cur.execute(insert_query, (
                person_name,
                state, # <-- 这里使用传入的 state 参数
                face_image_path,
                face_embedding_json_str
            ))
            new_person_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            print(f"\n成功将 '{person_name}' (ID: {new_person_id}, State: {state}) 的人脸信息保存到数据库。")
            return new_person_id
    except Exception as e:
        print(f"\n错误: 无法将人脸信息保存到数据库。详情: {e}")
        print("请检查数据库连接配置、表结构或是否存在违反唯一约束的情况（如 name 字段如果被设为唯一）。")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
    return None

cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print("错误: 无法打开摄像头。请确保摄像头连接正常，没有被其他程序占用。")
    sys.exit(1)

face_captured_success = False
person_name = input("请输入您要录入的姓名 (例如: 张三): ")

# 新增：询问用户 state
while True:
    try:
        person_state_input = input("请输入人员状态 (0: 非危险人物, 1: 危险人物): ")
        person_state = int(person_state_input)
        if person_state not in [0, 1]:
            print("输入无效。状态只能是 0 或 1。")
        else:
            break
    except ValueError:
        print("输入无效。请输入数字 0 或 1。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧。")
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 's' to capture, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Name: {person_name}, State: {person_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    detected_faces = enroll_worker.detect_faces(frame)
    if detected_faces:
        (x1, y1, x2, y2) = detected_faces[0]['box_coords']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
                    face_embedding_list = face_embedding.tolist()
                    
                    safe_person_name = person_name.strip().replace(" ", "_").lower()
                    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
                    image_path_for_db = f'/media/faces/{safe_person_name}_{timestamp}.jpg'

                    # 调用 save_person_to_db 时传入 person_state
                    if save_person_to_db(person_name, person_state, image_path_for_db, face_embedding_list):
                        face_captured_success = True
                        break
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