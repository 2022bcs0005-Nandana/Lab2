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
                    def validJson = readFile('valid_input.json').trim()
                    def response = sh(
                        script: """
                        docker run --rm --network jenkins-net curlimages/curl:latest \
                        curl -s -X POST http://wine_api_test:8000/predict \
                        -H 'Content-Type: application/json' \
                        -d '${validJson}'
                        """,
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
                    def invalidJson = readFile('invalid_input.json').trim()
                    def status = sh(
                        script: """
                        docker run --rm --network jenkins-net curlimages/curl:latest \
                        curl -s -o /dev/null -w '%{http_code}' \
                        -X POST http://wine_api_test:8000/predict \
                        -H 'Content-Type: application/json' \
                        -d '${invalidJson}'
                        """,
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
