pipeline {
    agent any

    environment {
        AWS_REGION = 'ap-south-1'
        ECR_REPO = 'patient-risk-app'
        EC2_USER = 'ubuntu'
        EC2_HOST = credentials('EC2_HOST')
        EC2_KEY = credentials('EC2_KEY')
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/ManoharSankar/Healthcare-Patient-Risk-Prediction-System.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Deploy Model to SageMaker') {
            steps {
                sh 'python train_model.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $ECR_REPO .'
            }
        }

        stage('Push to ECR') {
            steps {
                withAWS(region: "${AWS_REGION}", credentials: 'aws-creds') {
                    sh '''
                      $(aws ecr get-login --no-include-email --region $AWS_REGION)
                      docker tag $ECR_REPO:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/$ECR_REPO:latest
                      docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/$ECR_REPO:latest
                    '''
                }
            }
        }

        stage('Deploy to EC2') {
            steps {
                sh '''
                  echo "$EC2_KEY" > key.pem
                  chmod 600 key.pem
                  ssh -o StrictHostKeyChecking=no -i key.pem $EC2_USER@$EC2_HOST '
                    docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/$ECR_REPO:latest &&
                    docker stop patient-risk || true &&
                    docker rm patient-risk || true &&
                    docker run -d -p 8501:8501 --name patient-risk ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/$ECR_REPO:latest
                  '
                '''
            }
        }
    }
}
