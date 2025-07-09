from flask import Flask, request, jsonify
import time
import datetime

# 创建一个Flask应用
app = Flask(__name__)

# 定义一个只接受POST请求的路由，模拟接收计算任务
@app.route('/test-tunnel', methods=['POST'])
def test_tunnel():
    # 1. 打印日志，确认收到了请求，并记录时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] >>> 成功收到来自云服务器的隧道请求！")

    # 2. 尝试获取从云服务器Django应用发来的JSON数据
    try:
        data = request.json
        print(f"[{current_time}] >>> 收到的数据是: {data}")
    except Exception as e:
        print(f"[{current_time}] >>> 请求中没有有效的JSON数据: {e}")
        data = {}

    # 3. 模拟耗时的AI计算
    print(f"[{current_time}] >>> 正在模拟YOLO模型计算（等待3秒）...")
    time.sleep(3)
    print(f"[{current_time}] >>> 计算完成！")

    # 4. 构造并返回一个成功的JSON响应
    response_data = {
        'status': 'success',
        'message': '你好，云服务器！我是本地GPU机，我已成功处理你的请求。',
        'received_data': data,
        'model_result': [
            {'label': 'mock_person', 'confidence': 0.98, 'box': [10, 20, 100, 200]},
            {'label': 'mock_car', 'confidence': 0.95, 'box': [150, 250, 300, 400]}
        ]
    }
    return jsonify(response_data)

if __name__ == '__main__':
    print(">>> Mock AI服务器正在启动，监听地址 http://0.0.0.0:5000")
    print(">>> 等待通过SSH反向隧道传来的请求...")
    # 监听5000端口，允许所有IP访问
    app.run(host='0.0.0.0', port=5000)