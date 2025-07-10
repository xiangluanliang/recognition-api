# 人脸识别微服务 (recognition-api)

## 概览

本项目是“交通安防协同领航平台”中的**人脸识别微服务**。它是一个独立的 Flask 应用，负责视频/图像中的人脸检测、特征提取与识别，并将结果与 PostgreSQL 数据库交互（包括已知人脸比对和识别日志记录），并提供人脸注册接口。

## 核心功能

-   **人脸检测与特征提取**：从图像或视频帧中识别面部并生成唯一的数字特征向量。
-   **人脸识别**：将新特征与数据库中的已知人脸（`person` 表）进行比对，识别身份。
-   **识别日志**：记录所有成功的人脸识别事件到 `recognition_log` 表。
-   **人脸注册 API**：提供 Web 接口，通过图片将新人员信息录入 `person` 表。
-   **视频处理**：支持通过 URL 下载并处理视频文件。

## 架构定位

作为“智能安防公司”的“专业分析部门”，本服务从主 Django 应用接收任务，从流媒体服务器拉取视频，执行人脸 AI 分析，并将结构化结果（如识别事件）回传给主 Django 应用，同时直接与数据库交互。

## 设置与安装

### 1. 先决条件

-   Python 3.8+
-   Git
-   PostgreSQL 数据库实例
-   OpenCV DNN 模型文件：
    -   `opencv_face_detector.pbtxt`
    -   `opencv_face_detector_uint8.pb`
    -   `nn4.small2.v1.t7`
    将上述模型文件放入 `recognition-api/dnn_models/` 目录。

### 2. 环境设置

```bash
# 克隆仓库
git clone <your-repo-url>
cd recognition-api

# 创建并激活Python虚拟环境
python -m venv myvenv
source myvenv/Scripts/activate # Windows (Git Bash/MinGW64) / Linux/macOS
# 或者
# . myvenv/bin/activate # Linux/macOS 快捷方式

# 安装Python依赖
pip install -r requirements.txt
# 或手动安装: pip install Flask requests tqdm opencv-python numpy scipy psycopg2-binary
3. 服务配置 (环境变量)
重要：每次在新 Git Bash 会话中启动服务或工具前，必须设置以下环境变量。 生产环境中由部署系统管理。

Bash

export DB_HOST="你的PostgreSQL数据库公网IP或域名"
export DB_NAME="你的数据库名称" # 例如 rg_db
export DB_USER="你的数据库用户名" # 例如 rg_user
export DB_PASSWORD="你的数据库密码"
export DB_PORT="5432" # 如果端口不同，请修改
# 如果 Worker 需要回调 Django 主应用
# export DJANGO_MAIN_APP_BASE_URL="http://your_django_app_host:8000"
4. 数据库防火墙/安全组
确保云端 PostgreSQL 实例的防火墙/安全组已配置，允许你的开发机器的公共 IP 地址访问 PostgreSQL 端口（默认为 5432）。

使用指南 (API 接口)
启动 Worker 服务后，即可通过以下 API 接口进行交互。

1. 启动 Worker 服务
Bash

cd recognition-api/script
python video_processing_worker.py
服务将在 http://0.0.0.0:5000 监听。

2. /process-video (人脸识别)
分析视频或图片，进行人脸识别并记录日志。

方法: POST

输入 (JSON): {"video_url": "...", "camera_id": 1} 或 {"image_data": "base64...", "camera_id": 2}

输出 (JSON): 识别结果列表 (recognition_results), 触发警报 (alerts_triggered), status 等。

curl 示例 (本地视频):

Bash

# 确保在视频文件目录下运行 'python -m http.server 8000'
curl -X POST -H "Content-Type: application/json" -d '{ "video_url": "http://localhost:8000/test.mp4", "camera_id": 1 }' http://localhost:5000/process-video
3. /register-face (人脸注册)
通过图片将新人员信息注册到数据库。

方法: POST

输入 (JSON): {"image_data": "base64...", "person_name": "张三"}

输出 (JSON): 注册状态, 新生成的 person_id, face_embedding 等。

curl 示例:

Bash

# 使用 encode_image.py 获取 Base64 字符串
curl -X POST -H "Content-Type: application/json" -d '{ "image_data": "base64_string", "person_name": "李四" }' http://localhost:5000/register-face
4. /extract-embedding (提取特征向量)
从 Base64 图片中提取人脸特征向量。

方法: POST

输入 (JSON): {"image_data": "base64..."}

输出 (JSON): 提取的 face_embedding 列表, box_coords 等。

curl 示例:

Bash

curl -X POST -H "Content-Type: application/json" -d '{ "image_data": "base64_string" }' http://localhost:5000/extract-embedding
辅助工具
enroll_face.py (本地人脸注册工具)
通过本地摄像头捕获人脸，并直接注册到数据库（无需通过 API 接口）。

文件路径: recognition-api/enroll_face.py

使用方法:

Bash

# 确保已配置数据库环境变量
cd recognition-api
source myvenv/Scripts/activate
python enroll_face.py
按提示操作（例如，输入姓名，按 's' 捕捉）。

常见问题排查
TypeError: Object of type int64 is not JSON serializable:

原因：NumPy 特有数据类型未转换为 Python 原生类型。

解决：确保 video_processing_worker.py 为最新版本。

duplicate key value violates unique constraint "person_pkey":

原因：数据库 ID 序列与表中现有最大 ID 冲突。

解决：连接 PostgreSQL，执行 SELECT setval('person_id_seq', (SELECT MAX(id) FROM person) + 1, false);

Error: 403 Client Error: Forbidden for url: (视频下载失败):

原因：视频源有防盗链或访问限制。

解决：更换视频源，或使用本地 HTTP 服务器提供视频。

模型加载失败 (如 ...has no attribute 'FACE_DETECTOR_PROTOTXT_PATH'):

原因：模型文件路径错误或文件缺失。

解决：检查 recognition-api/dnn_models/ 目录，确保模型文件存在且路径正确。







