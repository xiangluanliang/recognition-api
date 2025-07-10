# api/views/video_views.py
import threading

from rest_framework.generics import RetrieveAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import requests
from ..models import IncidentDetectionLog, VideoAnalysisTask
from ..serializers import VideoAnalysisTaskSerializer
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


class VideoUploadAndProcessView(APIView):
    """
    接收视频上传，保存文件，创建任务记录，并触发后台AI处理。
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # 确保用户已登录
        if not request.user.is_authenticated:
            return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

        video_file = request.FILES.get('video')
        if not video_file:
            return Response({"error": "请求中未找到名为 'video' 的文件"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 使用Django的FileField来自动处理文件保存
            task = VideoAnalysisTask.objects.create(
                user=request.user,
                original_video=video_file,
                status=0  # 初始状态：等待处理
            )
            print(f"已创建新的视频分析任务，ID为: {task.id}")
        except Exception as e:
            return Response({"error": f"创建任务记录失败: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 构造视频的公网可访问URL，供AI Worker下载
        public_video_url = request.build_absolute_uri(task.original_video.url)

        # 2. 【关键】在这里调用你的my_yolo函数！
        # 我们使用多线程在后台调用，避免API请求超时
        print(f"准备启动后台线程来调用my_yolo处理任务 {task.id}...")
        thread = threading.Thread(
            target=my_yolo,
            args=(task.id, public_video_url)
        )
        thread.start()

        # 3. 立刻返回响应给前端
        print("已将任务交由后台处理，立刻返回响应给前端。")
        serializer = VideoAnalysisTaskSerializer(task)
        return Response(serializer.data, status=status.HTTP_202_ACCEPTED)

class TaskResultView(RetrieveAPIView):
    """
    根据任务ID获取单个视频分析任务的详细信息和状态。
    """
    queryset = VideoAnalysisTask.objects.all()
    serializer_class = VideoAnalysisTaskSerializer
    # lookup_field 告诉视图用URL中的哪个参数来查找对象，这里我们用'pk'或'id'
    # DRF默认就是'pk'，所以这行可以不写，但写上更清晰
    lookup_field = 'pk'