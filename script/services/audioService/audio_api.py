from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
import os

from .preprocess import extract_audio_from_video
from .event_handlers import handle_audio_file


class AudioDetectView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        if 'file' not in request.data:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        file_obj = request.data['file']
        temp_video_path = f"/tmp/{file_obj.name}"

        # 保存上传的视频到临时目录
        with open(temp_video_path, 'wb+') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        try:
            # 提取音频
            temp_audio_path = extract_audio_from_video(temp_video_path)

            # 用现有模型处理音频
            results = handle_audio_file(temp_audio_path)

            # 删除临时文件
            os.unlink(temp_video_path)
            os.unlink(temp_audio_path)

            return Response({"results": results})

        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
