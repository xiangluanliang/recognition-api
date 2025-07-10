# api/views/video_views.py
import threading

from rest_framework.generics import RetrieveAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import requests
from ..services.postService.send_yolo import my_yolo


# # 这是一个临时的后台任务函数，未来可以用Celery替代
# # 它负责与本地AI服务通信
# @shared_task
# def process_video_in_background(task_id, public_video_url):
#     """
#     通过反向代理将任务发送给本地AI服务，并更新数据库中的任务状态。
#     """
#     print(f"Celery Worker接收到任务：处理任务ID {task_id},视频URL: {public_video_url}")
#
#     # 1. 准备发往本地AI服务的指令
#     # 这是我们在服务器上测试过的隧道入口
#     tunnel_endpoint = "http://127.0.0.1:8001/process-video"
#     payload = {"video_url": public_video_url}
#
#     try:
#         # 2. 通过隧道发送请求 (设置一个较长的超时时间，例如5分钟)
#         response = requests.post(tunnel_endpoint, json=payload, timeout=300)
#         response.raise_for_status()  # 如果HTTP状态码是4xx或5xx，则抛出异常
#         ai_results = response.json()
#         print(f"从本地AI服务收到结果: {ai_results}")
#
#         # 3. 将结果更新到数据库
#         task_log = IncidentDetectionLog.objects.get(id=task_id)
#         # 假设 ai_results 包含 'status' 和 'confidence'
#         if ai_results.get('status') == 'success':
#             task_log.status = 1  # 1代表“处理完成”
#             task_log.confidence = ai_results.get('model_result', [{}])[0].get('confidence', 0.0)
#         else:
#             task_log.status = -1  # -1代表“处理失败”
#
#         task_log.save()
#         print(f"任务 {task_id} 已完成并更新数据库。")
#
#     except requests.exceptions.RequestException as e:
#         print(f"任务 {task_id} 处理失败: 与本地AI服务通信错误: {e}")
#         try:
#             task_log = IncidentDetectionLog.objects.get(id=task_id)
#             task_log.status = -1  # -1代表“处理失败”
#             task_log.save()
#         except IncidentDetectionLog.DoesNotExist:
#             print(f"无法找到任务ID {task_id} 来更新失败状态。")


