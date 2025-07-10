import os
import requests
from flask import Flask, request, jsonify
from tqdm import tqdm  # 用于显示漂亮的进度条
import datetime


# 创建Flask应用
app = Flask(__name__)

# 定义本地保存视频的目录
DOWNLOAD_DIR = "received_videos"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def download_video(url: str, directory: str) -> str | None:
    """根据URL下载视频文件到指定目录"""
    try:
        local_filename = os.path.join(directory, url.split('/')[-1])
        print(f"准备下载视频: {url}")
        print(f"将保存到: {local_filename}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f, tqdm(
                    desc=local_filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
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

    # 1. 获取指令中的视频URL
    data = request.json
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({'status': 'error', 'message': '请求中未提供 video_url'}), 400

    # 2. 从URL下载视频文件
    saved_path = download_video(video_url, DOWNLOAD_DIR)

    if saved_path:
        # 3. 在这里可以添加你的AI处理逻辑
        print(f"正在对 {saved_path} 进行AI分析（模拟）...")
        # ai_result = yolo_model.process(saved_path)

        # 4. 返回成功信息
        return jsonify({
            'status': 'success',
            'message': '视频已接收并保存成功',
            'saved_path_on_local_machine': saved_path
        })
    else:
        return jsonify({'status': 'error', 'message': '视频下载失败'}), 500


if __name__ == '__main__':
    print(">>> 视频处理Worker正在启动，监听地址 http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)