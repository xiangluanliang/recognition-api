# api/views.py
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from rest_framework.views import APIView
from rest_framework.response import Response
from api.models import TestNumber


class DoubleNumberView(APIView):
    # DRF会自动处理Token验证，如果验证通过，用户对象在request.user中
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        num_to_double = request.data.get('num')
        if num_to_double is None:
            return Response({"code": 400, "message": "缺少参数'num'"}, status=400)

        doubled_num = int(num_to_double) * 2

        TestNumber.objects.update_or_create(
            user=request.user,
            defaults={'number': doubled_num}
        )

        return Response({"code": 0, "info": {"num": doubled_num}})