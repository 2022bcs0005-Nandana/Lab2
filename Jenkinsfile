pipeline {
    agent any

    environment {
        IMAGE_NAME = "2022bcs0005/wine_predict_2022bcs0005:v02"
        CONTAINER_NAME = "wine_api_test"
        API_URL = "http://host.docker.internal:8000"
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
                docker ps -q --filter "publish=8000" | xargs -r docker rm -f
                docker run -d -p 8000:8000 --name wine_api_test 2022bcs0005/wine_predict_2022bcs0005:v02
                '''
            }
        }

        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 60, unit: 'SECONDS') {
                        waitUntil {
                            def status = sh(
                                script: "curl -s -o /dev/null -w \"%{http_code}\" ${API_URL}/docs",
                                returnStdout: true
                            ).trim()
                            echo "Service status: ${status}"
                            return status == "200"
                        }
                    }
                }
            }
        }

        stage('Send Valid Inference Request') {
            steps {
                script {
                    def response = sh(
                        script: "curl -s -X POST ${API_URL}/predict -H 'Content-Type: application/json' -d @valid_input.json",
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
                        script: "curl -s -o /dev/null -w \"%{http_code}\" -X POST ${API_URL}/predict -H 'Content-Type: application/json' -d @invalid_input.json",
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
            sh "docker rm -f ${CONTAINER_NAME} || true"
        }
        success {
            echo "PIPELINE SUCCESS"
        }
        failure {
            echo "PIPELINE FAILED"
        }
    }
}
