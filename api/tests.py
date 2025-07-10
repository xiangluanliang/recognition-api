from django.test import TestCase

# Create your tests here.
# api/tests.py
from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework.authtoken.models import Token
from rest_framework import status

class FeedbackAPITestCase(APITestCase):
    def setUp(self):
        """在每个测试方法运行前执行的设置"""
        # 1. 创建一个测试用户
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        # 2. 为该用户创建一个Token
        self.token = Token.objects.create(user=self.user)
        # 3. 设置API客户端，让后续请求都自动携带Token
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)

    def test_create_feedback(self):
        """测试创建一条新的反馈"""
        url = '/api/feedbacks/'
        data = {'title': '测试标题', 'content': '这是一条测试反馈内容。'}
        response = self.client.post(url, data, format='json')

        # 断言1：HTTP状态码应为201 (Created)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        # 断言2：数据库中应该确实增加了一条记录
        self.assertEqual(response.data['info']['title'], '测试标题')
        self.assertEqual(response.data['info']['user'], 'testuser')

    def test_list_feedbacks(self):
        """测试获取反馈列表"""
        # 先创建一条数据以便测试列表
        self.client.post('/api/feedbacks/', {'title': '测试标题', 'content': '内容'}, format='json')
        
        url = '/api/feedbacks/'
        response = self.client.get(url)

        # 断言1：HTTP状态码应为200 (OK)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # 断言2：返回的列表中应该有一条记录
        self.assertEqual(len(response.data['info']), 1)