# 文件名: mock_main_app.py
# 这个应用模拟了您的“主应用”中接收报警通知的接口。
# 当 video_processing_worker.py 发送报警时，它会在这里被接收并打印。

from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# 配置日志，以便在控制台清晰地看到接收到的报警
# (Mock Main App) 会显示在日志前缀，方便区分不同的应用输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (Mock Main App) - %(message)s')

@app.route('/api/alerts/receive/', methods=['POST'])
def receive_alert():
    """
    接收来自 video_processing_worker 的报警通知。
    这个接口会打印收到的报警数据到控制台。
    """
    try:
        # 获取 POST 请求的 JSON 数据
        alert_data = request.get_json()

        if not alert_data:
            app.logger.warning("收到空或无效的报警数据。")
            return jsonify({"status": "error", "message": "无效数据"}), 400

        # 从报警数据中提取关键信息，以方便打印
        alert_type = alert_data.get('alert_type', '未知类型')
        alert_level = alert_data.get('alert_level', '未知级别')
        camera_id = alert_data.get('camera_id', '未知摄像头')
        person_name = alert_data.get('person_name', 'N/A')
        person_id = alert_data.get('person_id', 'N/A')
        timestamp = alert_data.get('timestamp', 'N/A')

        # 构造要打印到控制台的详细报警信息
        log_message = (f"收到报警：类型='{alert_type}', 级别='{alert_level}', "
                       f"摄像头='{camera_id}', 人员='{person_name}' (ID:{person_id}), "
                       f"时间='{timestamp}'")
        
        # 根据报警级别，在控制台打印不同强调的信息，并模拟触发响应
        if alert_level == 'CRITICAL':
            app.logger.critical(log_message) # 使用 critical 级别日志，更醒目
            print("\n----- 【模拟报警系统：触发紧急响应！】 -----")
            print(f"收到紧急警报: 危险人物 {person_name} 在摄像头 {camera_id} 出现！")
            print("-------------------------------------------\n")
        elif alert_level == 'WARNING':
            app.logger.warning(log_message) # 使用 warning 级别日志
            print("\n----- 【模拟报警系统：发出警告！】 -----")
            print(f"收到警告: 陌生人 {person_name} 在摄像头 {camera_id} 出现！")
            print("---------------------------------------\n")
        else: # 对于 INFO 或 ERROR 级别的告警，只打印日志，不模拟特殊响应
            app.logger.info(log_message)

        # 返回成功响应给 video_processing_worker，告知报警已接收
        return jsonify({"status": "success", "message": "报警已接收"}), 200

    except Exception as e:
        app.logger.error(f"处理报警时发生错误: {e}")
        return jsonify({"status": "error", "message": "处理报警失败"}), 500

if __name__ == '__main__':
    # 这个模拟应用将运行在 8000 端口，与 video_processing_worker 的 5000 端口分开
    # 这样它们可以同时运行，互不干扰
    # debug=True 在开发阶段很有用，可以提供更多调试信息
    app.run(host='0.0.0.0', port=8000, debug=True)