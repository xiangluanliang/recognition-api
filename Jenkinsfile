pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Install Dependencies') {
            steps {
                sh "source /var/www/recognition-api/venv/bin/activate && pip install -r requirements.txt"
            }
        }
        stage('Migrate Database') {
            steps {
                script {
                    // 根据分支选择要使用的配置文件进行迁移
                    if (env.BRANCH_NAME == 'master') {
                        sh "source /var/www/recognition-api/venv/bin/activate && python manage.py migrate --settings=config.settings.production"
                    } else if (env.BRANCH_NAME == 'test') {
                        sh "source /var/www/recognition-api/venv/bin/activate && python manage.py migrate --settings=config.settings.test"
                    }
                }
            }
        }
        stage('Collect Static Files') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'master') {
                        sh "source /var/www/recognition-api/venv/bin/activate && python manage.py collectstatic --noinput --settings=config.settings.production"
                    } else if (env.BRANCH_NAME == 'test') {
                        sh "source /var/www/recognition-api/venv/bin/activate && python manage.py collectstatic --noinput --settings=config.settings.test"
                    }
                }
            }
        }
        stage('Restart Application') {
            steps {
                script {
                    // 根据分支重启对应的服务
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
}