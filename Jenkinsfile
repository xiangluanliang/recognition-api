pipeline {
    agent any

    environment {
        DEPLOY_DIR = (env.BRANCH_NAME == 'master') ? '/var/www/recognition-api-prod' : '/var/www/recognition-api-test'
        ENV_FILE = '.env'
    }

    stages {
        stage('Checkout') {
            steps {
                // 拉取代码到Jenkins的临时工作区
                checkout scm
            }
        }

        stage('Deploy Code') {
            steps {
                echo "正在将 ${env.BRANCH_NAME} 分支的代码同步到 ${env.DEPLOY_DIR}..."
                // 使用rsync将代码从工作区同步到分支对应的最终目录
                // --exclude会排除掉不必要的文件
                sh "rsync -av --delete --exclude='.git/' --exclude='venv/' --exclude='__pycache__/' ${WORKSPACE}/ ${env.DEPLOY_DIR}/"
            }
        }
        
        stage('Setup Environment and Dependencies') {
            steps {
                echo "在 ${env.DEPLOY_DIR} 中准备Python环境并安装依赖..."
                // 在一个sh块里执行，cd会保持在块内有效
                sh """
                    set -e
                    cd ${env.DEPLOY_DIR}
                    
                    # 如果虚拟环境目录不存在，则创建它
                    if [ ! -d "venv" ]; then
                        python3 -m venv venv
                    fi
                    
                    # 激活虚拟环境并安装依赖
                    # 保留了你使用的清华镜像源，非常好的实践！
                    . venv/bin/activate
                    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
                """
            }
        }

        stage('Run Django Commands') {
            steps {
                echo "为 ${env.BRANCH_NAME} 环境执行数据库迁移和静态文件收集..."
                sh """
                    set -e
                    cd ${env.DEPLOY_DIR}
                    . venv/bin/activate
                    
                    # 从对应的.env文件加载并导出环境变量
                    # grep -v '^#' 会过滤掉注释行
                    export \$(grep -v '^#' ${env.DEPLOY_DIR}/${env.ENV_FILE} | xargs)
                    
                    # 现在，环境变量已经设置好了，可以执行命令了
                    echo "执行数据库迁移..."
                    python manage.py migrate

                    echo "收集静态文件..."
                    python manage.py collectstatic --noinput
                """
            }
        }

        stage('Restart Application') {
            steps {
                script {
                    // 根据分支重启对应的Gunicorn服务
                    if (env.BRANCH_NAME == 'master') {
                        echo "重启 master 环境的Gunicorn服务 (gunicorn-prod)..."
                        sh "sudo systemctl restart gunicorn-prod"
                    } else if (env.BRANCH_NAME == 'test') {
                        echo "重启 test 环境的Gunicorn服务 (gunicorn-test)..."
                        sh "sudo systemctl restart gunicorn-test"
                    }
                }
            }
        }
    }
}