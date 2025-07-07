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