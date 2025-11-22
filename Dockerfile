# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend requirements file
COPY task-3/backend/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# We install this first to leverage Docker cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# We need task-1, task-2, and task-3 folders because the code relies on them
COPY task-1 /app/task-1
COPY task-2 /app/task-2
COPY task-3 /app/task-3

# Set the working directory to the backend folder for running the app
WORKDIR /app/task-3/backend

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable for port (optional, but good practice)
ENV PORT=8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

