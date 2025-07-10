pipeline {
    agent any

    stages {
        stage('Prepare Environment') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'master') {
                        env.DEPLOY_DIR = '/var/www/recognition-api-prod'
                    } else {
                        env.DEPLOY_DIR = '/var/www/recognition-api-test'
                    }
                    env.ENV_FILE = '.env'
                    echo "Branch: ${env.BRANCH_NAME}"
                    echo "Deploy Directory: ${env.DEPLOY_DIR}"
                    echo "Environment File: ${env.ENV_FILE}"
                }
            }
        }

        stage('Deploy Code') {
            steps {
                echo "正在将 ${env.BRANCH_NAME} 分支的代码同步到 ${env.DEPLOY_DIR}..."
                sh "rsync -av --delete --exclude='.git/' --exclude='venv/' --exclude='__pycache__/' ${WORKSPACE}/ ${env.DEPLOY_DIR}/"
            }
        }
        
        stage('Setup Python Environment & Dependencies') {
            steps {
                echo "在 ${env.DEPLOY_DIR} 中准备Python环境并安装依赖..."
                // 在一个sh块里执行，cd会保持在块内有效
                sh """
                    set -e
                    cd ${env.DEPLOY_DIR}
                    
                    if [ ! -d "venv" ]; then
                        python3 -m venv venv
                    fi
                    
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
                    
                    export \$(grep -v '^#' ${env.DEPLOY_DIR}/${env.ENV_FILE} | xargs)
                    
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
                    if (env.BRANCH_NAME == 'master') {
                        sh "sudo systemctl restart gunicorn-prod"
                    } else if (env.BRANCH_NAME == 'test') {
                        sh "sudo systemctl restart gunicorn-test"
                    }
                }
            }
        }
    }
}