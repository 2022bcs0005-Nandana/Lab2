pipeline {
    agent any
    environment {
        IMAGE_NAME = "2022bcs0005/wine_predict_2022bcs0005:v02"
        CONTAINER_NAME = "wine_api_test"
        API_URL = "http://wine_api_test:8000"
    }
    stages {
        stage('Pull Image') {
            steps {
                sh "docker pull ${IMAGE_NAME}"
            }
        }
        
        stage('Run Container') {
            steps {
                sh '''
                docker rm -f wine_api_test || true
                docker run -d --network jenkins-net --name wine_api_test 2022bcs0005/wine_predict_2022bcs0005:v02
                
                echo "Waiting for container to start..."
                sleep 10
                
                echo "=== Container Status ==="
                docker ps | grep wine_api_test
                
                echo "=== Container Logs ==="
                docker logs wine_api_test
                '''
            }
        }
        
        stage('Verify Test Files') {
            steps {
                sh '''
                echo "=== Checking for test files ==="
                ls -la *.json || echo "No JSON files found!"
                echo "=== Content of valid_input.json ==="
                cat valid_input.json || echo "File not found!"
                echo "=== Content of invalid_input.json ==="
                cat invalid_input.json || echo "File not found!"
                '''
            }
        }
        
        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 2, unit: 'MINUTES') {
                        waitUntil {
                            def result = sh(
                                script: '''
                                docker run --rm --network jenkins-net curlimages/curl:latest \
                                curl -s -o /dev/null -w %{http_code} http://wine_api_test:8000/docs
                                ''',
                                returnStdout: true
                            ).trim()
                            echo "Health check returned: ${result}"
                            return result == '200'
                        }
                    }
                }
            }
        }
        
        stage('Send Valid Inference Request') {
            steps {
                script {
                    def response = sh(
                        script: '''
                        docker run --rm --network jenkins-net -v /var/jenkins_home/workspace/2022BCS0005-Lab7:/workspace -w /workspace \
                        curlimages/curl:latest \
                        curl -s -X POST http://wine_api_test:8000/predict \
                        -H "Content-Type: application/json" \
                        --data-binary @valid_input.json
                        ''',
                        returnStdout: true
                    ).trim()
                    echo "Valid response: ${response}"
                    def json = readJSON text: response
                    if (!json.containsKey("wine_quality")) {
                        error("Prediction field missing")
                    }
                    if (!(json.wine_quality instanceof Number)) {
                        error("Prediction is not numeric")
                    }
                }
            }
        }
        
        stage('Send Invalid Request') {
            steps {
                script {
                    def status = sh(
                        script: '''
                        docker run --rm --network jenkins-net -v /var/jenkins_home/workspace/2022BCS0005-Lab7:/workspace -w /workspace \
                        curlimages/curl:latest \
                        curl -s -o /dev/null -w "%{http_code}" \
                        -X POST http://wine_api_test:8000/predict \
                        -H "Content-Type: application/json" \
                        --data-binary @invalid_input.json
                        ''',
                        returnStdout: true
                    ).trim()
                    echo "Invalid request status: ${status}"
                    if (status == "200") {
                        error("Invalid input was accepted")
                    }
                }
            }
        }
    }
    post {
        always {
            sh "docker logs ${CONTAINER_NAME} || true"
            sh "docker rm -f ${CONTAINER_NAME} || true"
        }
        success {
            echo "✅ PIPELINE SUCCESS"
        }
        failure {
            echo "❌ PIPELINE FAILED"
        }
    }
}
