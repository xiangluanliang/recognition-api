
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import (
    Subject
)
from ..serializers import (
    KnownFaceSerializer,EventLogCreateSerializer
)

class KnownFacesDataAPI(APIView):
    """
    为AI Worker提供所有已知人脸的特征数据。
    这个接口应该是受保护的，例如通过IP白名单或API密钥。
    为简化，这里暂时使用IsAuthenticated
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        # 从 "person" 表 (即Subject模型) 查询所有包含人脸特征的数据
        subjects = Subject.objects.filter(face_embedding__isnull=False)
        serializer = KnownFaceSerializer(subjects, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class LogEventAPI(APIView):
    """
    接收来自AI Worker的事件数据，并创建EventLog记录。
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = EventLogCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"status": "success", "message": "Event logged"}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)