# api/services/yoloService/yolo_service.py
import sys
import os
import django

sys.path.append('D:/project/recognition-api')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from flask import Flask, request, jsonify, send_from_directory
import datetime
from yolo_detector import detect_humans_with_yolov8
from yolo_abnormal_detector import detect_abnormal_behavior

app = Flask(__name__)
OUTPUT_DIR = "yolo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/detect-video', methods=['POST'])
def detect_video_api():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] === 收到目标检测请求 ===")

    data = request.json
    video_path = data.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': '视频路径无效'}), 400

    output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(video_path)}")
    success, msg, saved_path = detect_humans_with_yolov8(video_path, output_path)

    if success:
        return jsonify({
            'status': 'success',
            'message': msg,
            'video_url': f"/files/{os.path.basename(saved_path)}"
        })
    else:
        return jsonify({'status': 'error', 'message': msg}), 500

@app.route('/detect-abnormal', methods=['POST'])
def detect_abnormal_api():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] === 收到异常行为检测请求 ===")

    data = request.json
    video_path = data.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': '视频路径无效'}), 400

    output_path = os.path.join(OUTPUT_DIR, f"abnormal_{os.path.basename(video_path)}")
    success, msg, saved_path = detect_abnormal_behavior(video_path, output_path)

    if success:
        return jsonify({
            'status': 'success',
            'message': msg,
            'video_url': f"/files/{os.path.basename(saved_path)}"
        })
    else:
        return jsonify({'status': 'error', 'message': msg}), 500

# 提供视频文件访问（静态下载）
@app.route('/files/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    print(">>> YOLO检测服务已启动：监听 http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)