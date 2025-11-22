# Deployment Guide

This project is containerized using Docker and is designed to be deployed on **AWS App Runner** or **AWS ECS**.
Since you do not have a local Docker environment, the recommended deployment method is using **AWS App Runner** which builds the container image directly from your GitHub repository.

## Method 1: AWS App Runner (Recommended / Easiest)

AWS App Runner can connect to your GitHub repository, find the `Dockerfile` in the root, and build/deploy the application automatically.

### Steps:

1.  **Push Code to GitHub**:
    Ensure this entire project (root `Engage` folder) is pushed to a GitHub repository.

2.  **Go to AWS App Runner Console**:
    - Navigate to [AWS App Runner](https://console.aws.amazon.com/apprunner).
    - Click **Create service**.

3.  **Source & Deployment**:
    - **Repository type**: Source code repository.
    - **Provider**: GitHub (Connect your account if needed).
    - **Repository**: Select your repository.
    - **Branch**: Select `main` (or your working branch).
    - **Deployment settings**: "Automatic" (deploys on push) or "Manual".

4.  **Configure Build**:
    - **Configuration file**: "Configure all settings here".
    - **Runtime**: Docker (managed).
    - **Dockerfile path**: `Dockerfile` (default).
    - **Context directory**: `.` (default, root).

5.  **Configure Service**:
    - **Service name**: `engage-restaurant-api`.
    - **Port**: `8080`.
    - **Environment variables**: Add the following:
        - `GOOGLE_API_KEY`: Your Google Gemini API Key.
        - `AWS_REGION`: `us-east-1` (or your region).
        - `ENABLE_LOCAL_MODEL`: `true` (This runs the ML model inside the container).
        - `CHAT_CACHE_URL`: `memory://` (or a Redis URL if you have one).

6.  **Deploy**:
    - Review and click **Create & deploy**.
    - App Runner will pull the code, build the Docker image (installing all dependencies), and start the service.
    - Once active, you will get a **Default domain** (URL) to access your API.

## Method 2: AWS CloudShell (Build & Push to ECR)

If you need to push a specific image artifact or App Runner build fails, you can use **AWS CloudShell** (which has Docker installed) to build and push the image to Amazon ECR.

1.  **Open AWS CloudShell** in the Console.
2.  **Clone your repo**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO
    ```
3.  **Create ECR Repository**:
    ```bash
    aws ecr create-repository --repository-name engage-api
    ```
4.  **Login to ECR**:
    ```bash
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
    ```
5.  **Build & Push**:
    ```bash
    docker build -t engage-api .
    docker tag engage-api:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/engage-api:latest
    docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/engage-api:latest
    ```
6.  **Deploy**:
    - Use AWS App Runner (Image source) or AWS ECS to deploy this image.

## Architecture Notes

- **Monolith Container**: The RAG system (Task 1), ML Model (Task 2), and API (Task 3) are all bundled into a single container for simplicity.
- **Model Loading**: The `Dockerfile` copies `task-1` and `task-2` folders so the API can import them directly.
- **Persistence**: The ChromaDB SQLite file is baked into the image. For a production RAG system with updates, use a persistent volume (EFS) or a server-based vector DB (e.g., AWS OpenSearch or Chroma server).

