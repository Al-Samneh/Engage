# Deployment Script for SageMaker (Windows PowerShell)

# 1. Create the model artifact
Write-Host "Creating model.tar.gz..."

$ModelSource = "task-2\best_restaurant_rating_model_xgboost.pkl"
$DeployDir = "deploy\sagemaker"

# Ensure deploy dir exists
New-Item -ItemType Directory -Force -Path $DeployDir | Out-Null

# Copy model
Copy-Item -Path $ModelSource -Destination $DeployDir

Set-Location $DeployDir

Write-Host "Compressing artifact..."
# Use explicit relative paths with .\ prefix for PowerShell compatibility
# IMPORTANT: We are putting everything at the root of the tar file
# The structure will be:
#   best_restaurant_rating_model_xgboost.pkl
#   inference.py
#   requirements.txt
#   src/
#       ...
# This is what SageMaker expects when SAGEMAKER_SUBMIT_DIRECTORY is not set, or matches.
# By using -C code ., we are putting the CONTENTS of code/ at the root.
# And also adding the pickle file.

# 1. Copy pickle into code dir temporarily to make tarring easier
Copy-Item "best_restaurant_rating_model_xgboost.pkl" -Destination "code\"

# 2. Tar the contents of code/
tar -czvf model.tar.gz -C code .

# 3. Cleanup
Remove-Item "code\best_restaurant_rating_model_xgboost.pkl"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully created model.tar.gz"
} else {
    Write-Host "Error creating tarball. Make sure tar is in your PATH."
    exit 1
}

# Instructions for AWS
Write-Host "`n--- AWS Deployment Instructions ---"
Write-Host "1. Login to AWS CLI if not done:"
Write-Host "   aws configure"
Write-Host "2. Create an S3 bucket (if needed) and upload:"
Write-Host "   aws s3 cp model.tar.gz s3://YOUR_BUCKET_NAME/model.tar.gz"
Write-Host "3. Create SageMaker Model, Config, and Endpoint using the AWS Console or CLI."
Write-Host "   Image URI for sklearn: 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
Write-Host "   (Check standard AWS images for your region)"

# Cleanup copy of model
Remove-Item "best_restaurant_rating_model_xgboost.pkl"

Set-Location ..\..
