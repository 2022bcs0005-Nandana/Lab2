pipeline {
    agent any

    environment {
        ACCURACY = "0"
        DEPLOY = "false"
    }

    stages {

        // Stage 1: Checkout
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        // Stage 2: Setup Python Virtual Environment
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

        // Stage 3: Train Model
        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        // Stage 4: Read Accuracy
        stage('Read Accuracy') {
            steps {
                script {
                    sh '''
                    . venv/bin/activate
                    python - << 'EOF'
import json
with open("app/artifacts/metrics.json") as f:
    data = json.load(f)
print(data["accuracy"])
EOF
                    '''
                    def acc = sh(
                        script: ". venv/bin/activate && python -c \"import json;print(json.load(open('app/artifacts/metrics.json'))['accuracy'])\"",
                        returnStdout: true
                    ).trim()

                    env.ACCURACY = acc
                    echo "Current Accuracy: ${env.ACCURACY}"
                }
            }
        }

        // Stage 5: Compare Accuracy
        stage('Compare Accuracy') {
            steps {
                script {
                    def best = credentials('best-accuracy')
                    echo "Best Accuracy (stored): ${best}"

                    if (env.ACCURACY.toFloat() > best.toFloat()) {
                        env.DEPLOY = "true"
                    } else {
                        env.DEPLOY = "false"
                    }

                    echo "Deploy decision: ${env.DEPLOY}"
                }
            }
        }

        // Stage 6: Build Docker Image (Conditional)
        stage('Build Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub-creds') {
                        docker.build("2022bcs0005/wine_predict_2022bcs0005:${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        // Stage 7: Push Docker Image (Conditional)
        stage('Push Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub-creds') {
                        def app = docker.image("2022bcs0005/wine_predict_2022bcs0005:${env.BUILD_NUMBER}")
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
