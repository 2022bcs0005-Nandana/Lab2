pipeline {
    agent any

    environment {
        IMAGE_NAME = "2022bcs0005/wine_predict_2022bcs0005:v02"
        CONTAINER_NAME = "wine_api_test"
        API_URL = "http://localhost:8000"
    }

    stages {

        // Stage 1: Pull Image
        stage('Pull Image') {
            steps {
                sh "docker pull ${IMAGE_NAME}"
                sh "docker images | grep wine_predict_2022bcs0005"
            }
        }

        // Stage 2: Run Container
        stage('Run Container') {
            steps {
                sh "docker run -d --name ${CONTAINER_NAME} -p 8000:8000 ${IMAGE_NAME}"
            }
        }
        // Stage 3: Wait for Service Readiness
        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 60, unit: 'SECONDS') {
                        waitUntil {
                            def status = sh(
                                script: "curl -s -o /dev/null -w \"%{http_code}\" http://localhost:8000/docs",
                                returnStdout: true
                            ).trim()
                            echo "Service status: ${status}"
                            return (status == "200")
                        }
                    }
                }
            }
        }
        // Stage 4: Send Valid Inference Request
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

        // Stage 5: Send Invalid Request
        stage('Send Invalid Request') {
            steps {
                script {
                    def status = sh(
                        script: "curl -s -o /dev/null -w '%{http_code}' -X POST ${API_URL}/predict -H 'Content-Type: application/json' -d @invalid_input.json",
                        returnStdout: true
                    ).trim()

                    echo "Invalid request status: ${status}"

                    if (status == "200") {
                        error("Invalid input was accepted (should fail)")
                    }
                }
            }
        }

        // Stage 6: Stop Container
        stage('Stop Container') {
            steps {
                sh "docker stop ${CONTAINER_NAME} || true"
                sh "docker rm ${CONTAINER_NAME} || true"
            }
        }

        // Stage 7: Pipeline Result
        stage('Pipeline Result') {
            steps {
                echo "All API tests passed successfully."
            }
        }
    }

    post {
        always {
            sh "docker rm -f ${CONTAINER_NAME} || true"
        }
        success {
            echo "Pipeline SUCCESS: API validated correctly."
        }
        failure {
            echo "Pipeline FAILED: One or more validations failed."
        }
    }
}
