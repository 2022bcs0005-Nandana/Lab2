pipeline {
    agent any

    environment {
        ACCURACY = "0"
    }

    stages {

        stage('Checkout') {
            steps {
                git credentialsId: 'git-creds', url: 'https://github.com/2022bcs0005-Nandana/Lab'
            }
        }

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Read Accuracy') {
            steps {
                script {
                    def json = readFile('app/artifacts/metrics.json')
                    def parsed = new groovy.json.JsonSlurper().parseText(json)
                    env.ACCURACY = parsed.accuracy.toString()
                    echo "Current Accuracy: ${env.ACCURACY}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                script {
                    def best = credentials('best-accuracy')
                    echo "Best Accuracy: ${best}"

                    if (env.ACCURACY.toFloat() > best.toFloat()) {
                        env.DEPLOY = "true"
                    } else {
                        env.DEPLOY = "false"
                    }
                }
            }
        }

        stage('Build Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub-creds') {
                        def app = docker.build("2022bcs0005/wine_predict_2022bcs0005:${env.BUILD_NUMBER}")
                        app.push()
                        app.push("latest")
                    }
                }
            }
        }

        stage('Push Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                echo "Docker image pushed to Docker Hub"
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'app/artifacts/**', fingerprint: true
        }
    }
}
