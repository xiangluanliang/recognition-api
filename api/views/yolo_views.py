# api/views/yolo_views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import AllowAny
from api.services.yoloService.yolo_service import process_video
import base64
import cv2

class YoloDetectView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        video = request.FILES.get('video')
        if not video:
            return Response({"error": "未提供视频"}, status=400)

        frames = process_video(video)
        encoded_frames = []

        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            encoded = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(encoded)

        return Response({
            "code": 0,
            "frames": encoded_frames,
            "message": f"检测完成，共处理 {len(encoded_frames)} 帧"
        })
