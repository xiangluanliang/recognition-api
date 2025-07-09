pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo "拉取分支: ${env.BRANCH_NAME}"
                checkout scm
            }
        }

        stage('Install Dependencies') {
            stage('Install Dependencies') {
                steps {
                    echo "在虚拟环境中安装依赖..."
                    sh '''
                        . /var/www/recognition-api/venv/bin/activate
                        pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt \
                            -i https://pypi.tuna.tsinghua.edu.cn/simple \
                            --trusted-host pypi.tuna.tsinghua.edu.cn
                    '''
                }
            }
        }

        stage('Migrate Database') {
            steps {
                script {
                    // 根据分支名，选择要加载的环境文件
                    def envFile = (env.BRANCH_NAME == 'master') ? '.env.production' : '.env.test'
                    
                    // 在一个sh块里执行所有需要环境变量的命令
                    sh """
                        . /var/www/recognition-api/venv/bin/activate
                        
                        # 从对应的.env文件加载并导出环境变量
                        export \$(cat /var/www/recognition-api/${envFile} | xargs)
                        
                        # 现在，环境变量已经设置好了，可以执行命令了
                        echo "为 ${env.BRANCH_NAME} 分支执行数据库迁移..."
                        python manage.py migrate

                        echo "为 ${env.BRANCH_NAME} 分支收集静态文件..."
                        python manage.py collectstatic --noinput
                    """
                }
            }
        }

        stage('Restart Application') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'master') {
                        echo "重启 master 环境的Gunicorn服务..."
                        sh "sudo systemctl restart gunicorn-prod"
                    } else if (env.BRANCH_NAME == 'test') {
                        echo "重启 test 环境的Gunicorn服务..."
                        sh "sudo systemctl restart gunicorn-test"
                    }
                }
            }
        }
    }
}