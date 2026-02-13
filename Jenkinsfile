pipeline {
    agent any

    environment {
        ACCURACY = "0"
        DEPLOY = "false"
    }

    stages {

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
                    def acc = sh(
                        script: "python -c \"import json; print(json.load(open('app/artifacts/metrics.json'))['accuracy'])\"",
                        returnStdout: true
                    ).trim()

                    env.ACCURACY = acc
                    echo "Current Accuracy: ${env.ACCURACY}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_ACC')]) {
                    script {
                        echo "Best Accuracy: ${BEST_ACC}"

                        if (env.ACCURACY.toFloat() > BEST_ACC.toFloat()) {
                            env.DEPLOY = "true"
                        } else {
                            env.DEPLOY = "false"
                        }

                        echo "Deploy decision: ${env.DEPLOY}"
                    }
                }
            }
        }

        stage('Build and Push Docker Image') {
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
    }

    post {
        always {
            archiveArtifacts artifacts: 'app/artifacts/**', fingerprint: true
        }
    }
}
