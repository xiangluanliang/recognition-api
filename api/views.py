from rest_framework.views import APIView
from rest_framework.response import Response

class HelloWorldView(APIView):
    """
    一个简单的测试接口
    """
    def get(self, request):
        return Response({"message": "Hello, world! Backend is running."})