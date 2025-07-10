# video_processing_worker.py (运行在你的本地GPU电脑上)

import os
import requests
from flask import Flask, request, jsonify
from tqdm import tqdm
import datetime

# 1. 导入你自己的YOLO函数
#    请确保这个文件和你的YOLO函数在同一个目录下，或者yolo_script在Python路径中
# from yolo_detector import detect_humans_with_yolov8

# 创建Flask应用
app = Flask(__name__)

# 定义下载和处理后视频的保存目录
DOWNLOAD_DIR = "../media/received_videos"
PROCESSED_DIR = "../media/processed_videos"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_video(url: str, directory: str) -> str | None:
    try:
        local_filename = os.path.join(directory, url.split('/')[-1])
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f, tqdm(desc=f"下载 {local_filename}", total=total_size, unit='iB',
                                                       unit_scale=True, unit_divisor=1024) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print(f"'{local_filename}' 下载完成。")
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {url}, 错误: {e}")
        return None


# 定义一个路由，用于接收云服务器发来的处理指令
@app.route('/process-video', methods=['POST'])
def process_video_task():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] === 接到新的视频处理任务 ===")

    data = request.json
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({'status': 'error', 'message': '请求中未提供 video_url'}), 400

    # 1. 下载视频文件到本地
    downloaded_path = download_video(video_url, DOWNLOAD_DIR)

    if downloaded_path:
        try:
            # 2. 调用你真实的YOLO函数进行处理
            print(f"开始使用YOLO模型处理视频: {downloaded_path}")
            # 准备输出文件的路径
            base_filename = os.path.basename(downloaded_path)
            output_video_path = os.path.join(PROCESSED_DIR, f"processed_{base_filename}")

            # 假设你的函数会返回一个包含检测结果的列表/字典
            detection_results = [1,0,0,1] #detect_humans_with_yolov8(downloaded_path, output_video_path)

            print(f"YOLO处理完成，输出视频保存在: {output_video_path}")
            print(f"检测到的结果数据: {detection_results}")

            # 3. 返回包含真实检测结果的成功信息
            return jsonify({
                'status': 'success',
                'message': '视频分析成功',
                'detection_data': detection_results,
                'processed_video_path_on_worker': output_video_path  # 这个路径仅供本地参考
            })

        except Exception as e:
            print(f"YOLO函数执行出错: {e}")
            return jsonify({'status': 'error', 'message': f'AI模型处理失败: {e}'}), 500
    else:
        return jsonify({'status': 'error', 'message': '视频下载失败'}), 500


if __name__ == '__main__':
    print(">>> 真实AI Worker正在启动，监听地址 http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)