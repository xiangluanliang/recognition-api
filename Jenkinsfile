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
            steps {
                echo "在虚拟环境中安装依赖..."
                // 使用 . 代替 source
                sh ". /var/www/recognition-api/venv/bin/activate && pip install -r requirements.txt"
            }
        }

        stage('Migrate Database') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'master') {
                        echo "为 master 分支执行数据库迁移..."
                        // 使用 . 代替 source
                        sh ". /var/www/recognition-api/venv/bin/activate && python manage.py migrate --settings=config.settings.production"
                    } else if (env.BRANCH_NAME == 'test') {
                        echo "为 test 分支执行数据库迁移..."
                        // 使用 . 代替 source
                        sh ". /var/www/recognition-api/venv/bin/activate && python manage.py migrate --settings=config.settings.test"
                    }
                }
            }
        }
        
        stage('Collect Static Files') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'master') {
                        echo "为 master 分支收集静态文件..."
                        // 使用 . 代替 source
                        sh ". /var/www/recognition-api/venv/bin/activate && python manage.py collectstatic --noinput --settings=config.settings.production"
                    } else if (env.BRANCH_NAME == 'test') {
                        echo "为 test 分支收集静态文件..."
                        // 使用 . 代替 source
                        sh ". /var/www/recognition-api/venv/bin/activate && python manage.py collectstatic --noinput --settings=config.settings.test"
                    }
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