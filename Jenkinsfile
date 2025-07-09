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
                sh "source /var/www/recognition-api/venv/bin/activate && python manage.py migrate"
            }
        }
        stage('Restart Application') {
            steps {
                sh "sudo systemctl restart gunicorn"
            }
        }
    }
}