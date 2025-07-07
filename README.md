# 识别项目后端

## 项目介绍

## 初始化配置

### 环境要求 (Prerequisites)

在开始之前，请确保您的开发环境中已安装以下软件。

  - **Python**: **3.11+**

      - **检查版本**: 在终端运行 `python --version` 或 `python3 --version`
      - **下载地址**: [https://www.python.org/](https://www.python.org/)

### 初始化流程 (Initialization Steps)

1. 克隆项目代码

```bash
git clone git@github.com:xiangluanliang/recognition-client.git
```

2. 创建并激活虚拟环境

在项目根目录的终端中，运行以下命令。这会创建一个名为 `myvenv` 的虚拟环境文件夹。

```bash
python -m venv myvenv
```

*注意：根据您的系统配置，您可能需要使用 `python3` 替代 `python`*

3.  激活虚拟环境

激活后，您将在终端提示符的开头看到 `(venv)` 字样。

```bash
venv\Scripts\activate
```

4. 创建或检查 `requirements.txt` 文件

`requirements.txt` 文件定义了项目所需的所有Python包及其版本。请在项目根目录下创建（或检查）此文件，并确保其包含以下内容，以保证团队成员环境一致：

```txt
asgiref==3.9.0
Django==5.2.4
django-cors-headers==4.7.0
djangorestframework==3.16.0
drf-yasg==1.21.10
inflection==0.5.1
packaging==25.0
pytz==2025.2
PyYAML==6.0.2
sqlparse==0.5.3
tzdata==2025.2
uritemplate==4.2.0
```

**提示**: 如果您后续安装了新的包，可以使用 `pip freeze > requirements.txt` 命令来更新此文件。

5. 使用 `requirements.txt` 安装所有依赖

在**已激活虚拟环境**的终端中，运行以下命令。pip会自动读取文件列表并安装所有必需的包。

```bash
pip install -r requirements.txt
```

6. 初始化数据库

在首次运行项目或数据库结构有变更时，需要执行数据库迁移。

```bash
# 执行数据库迁移，这将根据代码模型创建数据库表
python manage.py migrate
```

此命令会根据项目默认配置在根目录创建一个 `db.sqlite3` 数据库文件。

7. 运行项目 (Running the Project)

完成以上所有步骤后，您的后端环境已准备就绪。执行以下命令启动开发服务器:

```bash
python manage.py runserver
```

服务器默认运行在 `http://127.0.0.1:8000/`。您可以通过访问以下地址来验证项目是否成功运行：

- **API文档 (Swagger)**: [http://127.0.0.1:8000/swagger/](https://www.google.com/search?q=http://127.0.0.1:8000/swagger/)
- **测试API接口**: [http://127.0.0.1:8000/api/hello/](https://www.google.com/search?q=http://127.0.0.1:8000/api/hello/)
- **Django管理后台**: [http://127.0.0.1:8000/admin/](https://www.google.com/search?q=http://127.0.0.1:8000/admin/)

如果以上地址都能正常访问，则说明您的后端开发环境已成功搭建。

## 添加一个完整功能接口的标准化流程

本部分旨在为后端开发团队提供一套清晰、规范的接口开发流程。遵循此流程将确保我们的代码结构清晰、可维护性高、质量可靠且易于测试。文档以**用户反馈**为例。

**功能目标**：允许已登录用户提交反馈信息（包含标题和内容），并能查看所有反馈列表。

### 第一步：定义数据模型 (`models.py`) - "数据蓝图"

**目标**：确定我们需要存储什么数据，并在数据库中为创建表结构。

1.  **编辑模型文件**：打开 `api/models.py` 文件，添加 `Feedback` 模型。

    ```python
    # api/models.py
    from django.db import models
    from django.contrib.auth.models import User

    # ... 其他已有的模型 ...

    class Feedback(models.Model):
        # 关联提交反馈的用户，一个用户可以有多个反馈
        user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
        # 反馈标题
        title = models.CharField(max_length=100)
        # 反馈内容
        content = models.TextField()
        # 创建时间，在创建记录时自动填充当前时间
        created_at = models.DateTimeField(auto_now_add=True)

        def __str__(self):
            return f"'{self.title}' by {self.user.username}"
    ```

2.  **更新数据库**：在终端中（虚拟环境需激活）运行以下命令，将模型应用到数据库。

    ```bash
    # 1. 创建新的迁移文件
    python manage.py makemigrations api

    # 2. 将迁移应用到数据库
    python manage.py migrate
    ```

    至此，数据库中已经有了一张名为 `api_feedback` 的新表。

-----

### 第二步：在Admin后台中注册模型 (`admin.py`) - "后台管理"

**目标**：让开发者和管理员能在一个友好的网页界面上方便地查看和管理新功能的数据。

1.  **编辑Admin文件**：打开 `api/admin.py`，注册 `Feedback` 模型。

    ```python
    # api/admin.py
    from django.contrib import admin
    from .models import TestNumber, Feedback # 导入Feedback模型

    admin.site.register(TestNumber)
    admin.site.register(Feedback) # 注册Feedback模型
    ```

2.  **可视化验证**：

      * 确保您的开发服务器正在运行 (`python manage.py runserver`)。
      * 打开浏览器访问 [http://127.0.0.1:8000/admin/](https://www.google.com/search?q=http://127.0.0.1:8000/admin/)。
      * 使用您创建的超级管理员账户登录。
      * 您现在应该能在后台主页看到一个新的\*\*"Feedbacks"\*\*模块，您可以点击进入并手动添加或查看数据。

-----

### 第三步：创建数据序列化器 (`serializers.py`) - "数据翻译官"

**目标**：定义API接口的数据格式，负责将数据库中的数据安全地翻译成前端需要的JSON格式。

> 在DRF中，几乎每一个API视图（View）都应该搭配一个对应的Serializer。它能强制您将数据格式、验证逻辑和业务逻辑分离开来，让您的代码更安全、更清晰、也更易于维护。

1.  **编辑Serializer文件**：打开 `api/serializers.py`，添加 `FeedbackSerializer`。

    ```python
    # api/serializers.py
    from rest_framework import serializers
    from django.contrib.auth.models import User
    from .models import Feedback # 导入Feedback模型

    # ... UserSerializer ...

    class FeedbackSerializer(serializers.ModelSerializer):
        # 让返回的JSON中包含用户名，而不仅仅是用户ID
        user = serializers.CharField(source='user.username', read_only=True)

        class Meta:
            model = Feedback
            # 定义API应该暴露哪些字段
            fields = ['id', 'user', 'title', 'content', 'created_at']
            # 将user字段设为只读，因为我们会根据当前登录用户自动设置
            read_only_fields = ['user', 'created_at']
    ```

-----

### 第四步：编写视图和方法 (`views.py`) - "业务逻辑调度"

**目标**：创建处理API请求的核心逻辑。

> 并不是每一个新方法都需要定义一个新视图类。视图类的核心职责是处理HTTP请求和响应，它应该像一个“调度员”，而不是“技术专家”。
> 前期开发时，如无必要，建议尽量把新需求写在现有类里，只有当一个类显得过于臃肿，或者其他必要情况时才需要分出一个新类。本例把FeedbackView类直接写在data_views.py后面其实就可以了，没必要创建一个新文件，这里仅为展示而做。
> 如果有图像识别这样的业务，直接把方法写在`services.py`里，它接收图像数据，执行复杂的识别逻辑，然后返回结果。`services`中的函数完全不关心HTTP请求或响应，可以像写任何算法作业的函数一样去写它。但本例不涉及这个模块。
> 同样的，`services.py`在后期可能也会很臃肿，可以到时候再像`views`一样把它拆成包。

1.  **编辑视图文件**：打开 `api/views`，添加 `feedback_views.py`类。这个类将处理所有关于反馈的请求。

    ```python
    # api/views/feedback_views.py
    from ..models import Feedback # 导入Feedback模型
    from ..serializers import FeedbackSerializer # 导入FeedbackSerializer
    # from .. import services # 根据具体需求

    # ... 其他视图 ...

    class FeedbackView(APIView):
        authentication_classes = [TokenAuthentication]
        permission_classes = [IsAuthenticated]

        # 定义GET方法，用于获取反馈列表
        def get(self, request):
            """获取所有反馈列表"""
            feedbacks = Feedback.objects.all().order_by('-created_at')
            serializer = FeedbackSerializer(feedbacks, many=True) # many=True表示序列化的是一个列表
            return Response({"code": 0, "info": serializer.data})

        # 定义POST方法，用于创建一条新反馈
        def post(self, request):
            """创建一条新反馈"""
            serializer = FeedbackSerializer(data=request.data)
            if serializer.is_valid(raise_exception=True): # 验证前端发来的数据是否合法
                # 将当前登录的用户与反馈关联起来，然后保存到数据库
                serializer.save(user=request.user)
                return Response({"code": 0, "info": serializer.data}, status=201)
            # 如果is_valid验证失败且没有raise_exception，则会执行下面这行
            # return Response({"code": 400, "message": "输入数据无效", "details": serializer.errors}, status=400)
    ```

-----

### 第五步：添加接口路由 (`urls.py`) - "创建API地址"

**目标**：为我们新创建的视图分配一个独一无二的URL地址。

1.  **编辑路由文件**：打开 `api/urls.py`，添加新路径。

    ```python
    # api/urls.py
    from .views import HelloWorldView, DoubleNumberView, UserDetailView, FeedbackView # 导入FeedbackView

    urlpatterns = [
        # ... 其他路由 ...
        path('users/<int:user_id>/', UserDetailView.as_view(), name='user_detail'),
        
        # 添加用于处理反馈的路由
        path('feedbacks/', FeedbackView.as_view(), name='feedback-list-create'),
    ]
    ```

-----

### 第六步：编写测试用例 (`tests.py`) - "质量保证"

**目标**：编写自动化测试，确保我们的新接口能正常工作，并且在未来不会被意外破坏。

1.  **编辑测试文件**：打开 `api/tests.py`，添加测试用例。

    ```python
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
    ```

2.  **运行测试**：在终端中运行。

    ```bash
    python manage.py test api
    ```

    如果看到 `OK`，说明您的测试全部通过，新接口质量有保证。

-----

### 第七步：可视化地查看与测试结果

**目标**：通过Web界面直观地确认我们的工作成果。

1.  **在Admin后台查看**：

      * 登录到 [http://127.0.0.1:8000/admin/](https://www.google.com/search?q=http://127.0.0.1:8000/admin/)。
      * 进入 **Feedbacks** 模块。
      * 您应该能看到刚才在 `test_create_feedback` 测试中自动创建的那条“测试标题”的反馈记录。

2.  **在Swagger UI中交互式测试**：

      * 打开API文档 [http://127.0.0.1:8000/swagger/](https://www.google.com/search?q=http://127.0.0.1:8000/swagger/)。
      * 展开新的 `/api/feedbacks/` 接口。
      * **测试POST**:
          * 点击POST方法的“Try it out”。
          * 在请求体中输入 `{"title": "从Swagger发送", "content": "内容..."}`。
          * 点击“Execute”按钮。您应该能看到成功的响应。
      * **测试GET**:
          * 点击GET方法的“Try it out”。
          * 点击“Execute”按钮。您应该能看到一个包含所有反馈的列表。

-----

通过遵循这套完整的流程，您不仅添加了功能，还保证了它的可管理性、可测试性和文档的清晰性，这是专业后端开发的标准路径。