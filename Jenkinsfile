pipeline {
    agent any

    environment {
        IMAGE_NAME = "2022bcs0005/wine_predict_2022bcs0005:v02"
        CONTAINER_NAME = "wine_api_test"
    }

    stages {

        stage('Pull Image') {
            steps {
                sh "docker pull ${IMAGE_NAME}"
                sh "docker images | grep wine_predict_2022bcs0005"
            }
        }

        stage('Run Container') {
            steps {
                sh '''
                docker rm -f wine_api_test || true
                docker run -d --name wine_api_test ${IMAGE_NAME}
                '''
            }
        }

        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 1, unit: 'MINUTES') {
                        waitUntil {
                            def status = sh(
                                script: '''
docker exec wine_api_test python - <<EOF
import requests
try:
    r = requests.get("http://localhost:8000/docs", timeout=2)
    print(r.status_code)
except:
    print("000")
EOF
                                ''',
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
                        script: '''
docker exec wine_api_test python - <<EOF
import requests, json
data = json.load(open("valid_input.json"))
r = requests.post("http://localhost:8000/predict", json=data)
print(r.text)
EOF
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
docker exec wine_api_test python - <<EOF
import requests, json
data = json.load(open("invalid_input.json"))
r = requests.post("http://localhost:8000/predict", json=data)
print(r.status_code)
EOF
                        ''',
                        returnStdout: true
                    ).trim()

                    echo "Invalid request status: ${status}"

                    if (status == "200") {
                        error("Invalid input was accepted (should fail)")
                    }
                }
            }
        }

        stage('Stop Container') {
            steps {
                sh "docker rm -f ${CONTAINER_NAME} || true"
            }
        }

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
            echo "Pipeline SUCCESS"
        }
        failure {
            echo "Pipeline FAILED"
        }
    }
}
