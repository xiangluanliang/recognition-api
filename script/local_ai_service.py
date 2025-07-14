# local_ai_service.py
import base64
import os
import cv2
import datetime

import numpy as np
import requests
import threading
import logging
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from yolo_detector import process_single_frame as person_detector
from face_recognizer2 import process_frame_face_recognition
from report_generator import process_report_generation

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (AI-Worker) - %(message)s')

AI_FUNCTIONS = {
    'person_detection': person_detector,
    'face_recognition': process_frame_face_recognition,
    # 未来有新功能，在这里继续添加...
}

DJANGO_API_TOKEN = os.environ.get('DJANGO_API_TOKEN', '3d814802906b91d7947518f5d0191a42795cace7')
DJANGO_API_BASE_URL = os.environ.get('DJANGO_API_URL', 'http://172.21.167.220:8000/api/test/')
RTMP_SERVER_URL = os.environ.get('RTMP_SERVER_URL', 'rtmp://localhost:9090/live')
KNOWN_FACES_CACHE = []
CACHE_REFRESH_INTERVAL = 300


def fetch_and_cache_known_faces():
    global KNOWN_FACES_CACHE
    try:
        url = os.path.join(DJANGO_API_BASE_URL, 'known-faces/')
        response = requests.get(url, timeout=10, headers={"Authorization": f"Token {DJANGO_API_TOKEN}"})
        response.raise_for_status()
        KNOWN_FACES_CACHE = response.json()
        app.logger.info(f"Successfully refreshed known faces cache. Loaded {len(KNOWN_FACES_CACHE)} faces.")
    except Exception as e:
        app.logger.error(f"Failed to fetch known faces from Django: {e}")


def schedule_face_cache_refresh():
    fetch_and_cache_known_faces()
    threading.Timer(CACHE_REFRESH_INTERVAL, schedule_face_cache_refresh).start()


def log_event_to_django(event_data: dict):
    def _send_request():
        try:
            url = os.path.join(DJANGO_API_BASE_URL, 'log-event/')
            requests.post(url, json=event_data, timeout=5, headers={"Authorization": f"Token {DJANGO_API_TOKEN}"})
        except Exception as e:
            app.logger.error(f"Failed to log event to Django: {e}")

    threading.Thread(target=_send_request).start()


def stream_processor(ai_function_name: str, stream_id: str):
    """
    动态地调用AI函数，并智能地处理其特有的参数和返回结果。
    """
    process_function = AI_FUNCTIONS.get(ai_function_name)
    if not process_function:
        app.logger.error(f"Unknown AI function requested: {ai_function_name}")
        return

    rtmp_url = f'{RTMP_SERVER_URL}{stream_id}'
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        app.logger.error(f"Cannot open stream: {rtmp_url}")
        return

    app.logger.info(f"Processing stream '{stream_id}' with AI function '{ai_function_name}'")

    frame_skip = 5
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success: break

        processed_frame, detection_data = None, None

        if frame_count % frame_skip == 0:
            if ai_function_name == 'face_recognition':
                processed_frame, detection_data = process_function(
                    frame,
                    known_faces_data=KNOWN_FACES_CACHE,
                    camera_id=int(stream_id)
                )
            else:
                processed_frame, detection_data = process_function(
                    frame,
                    camera_id=int(stream_id)
                )

            if detection_data and 'events_to_log' in detection_data:
                for event in detection_data['events_to_log']:
                    event['camera'] = int(stream_id)
                    event['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                    log_event_to_django(event)

        else:
            processed_frame = frame

        if processed_frame is None: processed_frame = frame

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        frame_count += 1
    cap.release()


@app.route('/<ai_function_name>/<stream_id>')
def video_feed(ai_function_name: str, stream_id: str):
    return Response(
        stream_processor(ai_function_name, stream_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/ai/recognize-frame', methods=['POST'])
def recognize_frame_api():
    data = request.json
    image_base64 = data.get('image_data')

    if not image_base64:
        return jsonify({'status': 'error', 'message': '未提供 image_data'}), 400

    try:
        # 解码Base64图片
        # 前端发送的 toDataURL() 结果会包含一个头部 "data:image/jpeg;base64,"，需要去掉
        if ',' in image_base64:
            header, encoded = image_base64.split(',', 1)
        else:
            encoded = image_base64

        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("图像解码失败")
        else:
            app.logger.info("图像解码成功")

    except Exception as e:
        app.logger.error(f"Base64 image decoding failed: {e}")
        return jsonify({'status': 'error', 'message': '无效的图像数据'}), 400

    # --- 调用我们标准的人脸识别功能 ---
    # 注意：这里我们硬编码了使用 'face_recognition' 功能
    # 第二个参数是缓存的已知人脸数据，由后台定时任务更新
    processed_frame, detection_data = process_frame_face_recognition(
        frame,
        known_faces_data=KNOWN_FACES_CACHE,
        camera_id=0  # 身份认证场景通常用0或一个虚拟ID
    )

    # 1. 将处理后的图像帧(Numpy数组)编码为JPEG格式
    _, buffer = cv2.imencode('.jpg', processed_frame)
    # 2. 将JPEG数据转换为Base64字符串
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
    # 3. 将Base64字符串添加到要返回的字典中
    detection_data['processed_image'] = f"data:image/jpeg;base64,{processed_image_base64}"

    # 4. 返回包含所有数据（JSON信息+处理后图像）的响应
    return jsonify(detection_data)


def fetch_summary_for_report():
    """从Django后端获取日报所需的数据摘要。"""
    # 这个函数现在只负责调用API
    api_url = f"{DJANGO_API_BASE_URL}daily-report/today"
    headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
    app.logger.info(f"ReportGen: Fetching data from {api_url}")
    response = requests.get(api_url, headers=headers, timeout=20, verify=False)
    response.raise_for_status()
    return response.json()


@app.route('/ai/generate-report', methods=['POST'])
def generate_report_endpoint():
    """
    API端点，调用独立的AI模块来生成日报。
    """
    app.logger.info("Received request to generate daily report.")
    try:
        # 1. 从Django获取数据
        summary_data = fetch_summary_for_report()

        # 2. 【核心修改】调用外部模块的函数进行AI处理
        report_content = process_report_generation(summary_data)

        return jsonify({
            "status": "success",
            "message": "Daily report generated and submitted successfully.",
            "content": report_content
        }), 200

    except Exception as e:
        app.logger.error(f"Failed to execute report generation task: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.logger.info(">>> Unified AI Worker Service Starting <<<")
    app.logger.info(f">>> Registered real-time AI functions: {list(AI_FUNCTIONS.keys())}")
    if not DJANGO_API_TOKEN:
        app.logger.warning(
            "CRITICAL: DJANGO_API_TOKEN environment variable is not set. Service may not function correctly.")

    app.logger.info(f">>> Django API URL set to: {DJANGO_API_BASE_URL}")
    app.logger.info(f">>> RTMP Stream Source URL set to: {RTMP_SERVER_URL}")
    app.logger.info(">>> Report generation module is also loaded.")
    # 启动时先执行一次，然后开启定时刷新
    if 'face_recognition' in AI_FUNCTIONS:
        threading.Thread(target=schedule_face_cache_refresh, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)