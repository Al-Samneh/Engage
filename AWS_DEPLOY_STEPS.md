# Deployment Guide: Engage Restaurant AI

This guide covers the deployment of the **SageMaker Model** (Task 2) and the **FastAPI Backend** (Task 3).

## Prerequisites

1.  **AWS Account** with permissions for SageMaker, S3, and App Runner.
2.  **AWS CLI** installed and configured (`aws configure`).
3.  **GitHub Repository** with this code pushed.

---

## Part 1: Deploy ML Model to SageMaker

We will deploy the rating prediction model (`best_restaurant_rating_model_xgboost.pkl`) as a serverless endpoint.

### Step 1: Upload Model Artifact
The script `deploy_sagemaker.ps1` created a `model.tar.gz` file. Upload it to S3.

```bash
# Replace YOUR_BUCKET_NAME with a unique name (e.g., engage-models-2025)
aws s3 mb s3://YOUR_BUCKET_NAME
aws s3 cp deploy/sagemaker/model.tar.gz s3://YOUR_BUCKET_NAME/model.tar.gz
```

### Step 2: Create IAM Role
You need an IAM role that allows SageMaker to read from S3.
1. Go to **IAM Console** > **Roles** > **Create role**.
2. Service: **SageMaker**.
3. Permissions: `AmazonSageMakerFullAccess` (for testing).
4. Name: `EngageSageMakerRole`.
5. Copy the **Role ARN** (starts with `arn:aws:iam::...`).

### Step 3: Create SageMaker Model
Register the model in SageMaker, pointing to your S3 file.

```bash
aws sagemaker create-model \
    --model-name engage-rating-model \
    --primary-container Image=683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3,ModelDataUrl=s3://YOUR_BUCKET_NAME/model.tar.gz \
    --execution-role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/EngageSageMakerRole
```

### Step 4: Create Endpoint Config
Define the hardware (instance type).

```bash
aws sagemaker create-endpoint-config \
    --endpoint-config-name engage-rating-config \
    --production-variants VariantName=AllTraffic,ModelName=engage-rating-model,InitialInstanceCount=1,InstanceType=ml.t2.medium
```

### Step 5: Create Endpoint
Launch the actual server.

```bash
aws sagemaker create-endpoint \
    --endpoint-name engage-rating-endpoint \
    --endpoint-config-name engage-rating-config
```

*Wait ~5-10 minutes for status to change from `Creating` to `InService`.*

---

## Part 2: Deploy API to AWS App Runner

The API handles the RAG search and acts as a gateway to the SageMaker model.

### Step 1: Configure App Runner
1. Go to **AWS App Runner Console** > **Create service**.
2. **Source**: Source code repository > Connect your GitHub repo.
3. **Branch**: `main`.
4. **Deployment settings**: Automatic.

### Step 2: Build Settings
- **Runtime**: Docker (Managed).
- **Dockerfile path**: `Dockerfile` (root).
- **Port**: `8080`.

### Step 3: Environment Variables
Add these key-value pairs:

| Key | Value | Description |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | `AIzaSy...` | Your Gemini API Key |
| `AWS_REGION` | `us-east-1` | Your AWS Region |
| `ENABLE_LOCAL_MODEL` | `false` | Disable embedded model to use SageMaker |
| `SAGEMAKER_ENDPOINT_NAME` | `engage-rating-endpoint` | The name from Part 1, Step 5 |

### Step 4: Deploy
Click **Create & deploy**. App Runner will build the container and provide a public URL (e.g., `https://xyz.awsapprunner.com`).

---

## Verification

1. **Health Check**: `GET /v1/health` -> Should show `status: ok`.
2. **Search**: `POST /v1/search` -> Test RAG functionality.
3. **Rating**: `POST /v1/rating/predict` -> Test SageMaker integration.

