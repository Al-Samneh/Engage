# Engage - Restaurant Intelligence Platform

Engage is a comprehensive AI-powered platform for restaurant discovery and analytics. It combines a RAG-based conversational search assistant with a sophisticated machine learning model for predicting user ratings.

## Project Structure

The repository is organized into three main tasks:

### 1. RAG Search Assistant (`task-1`)
A conversational agent that helps users find restaurants using natural language.
- **Tech Stack:** LangGraph, LangChain, ChromaDB, Google Gemini (Flash Lite).
- **Key Features:**
  - **Metadata Extraction:** Converts natural language into structured filters (Cuisine, Location, Price).
  - **Context Awareness:** Maintains conversation history and intelligently handles context resets.
  - **Hybrid Search:** Combines vector semantic search with metadata filtering for high precision.

### 2. Rating Prediction Model (`task-2`)
A machine learning pipeline that predicts the star rating a user would give to a restaurant.
- **Tech Stack:** XGBoost, Scikit-learn, Sentence-Transformers.
- **Methodology:**
  - **Hybrid Features:** Uses structured data (price, location), user profiles, time-series trends, and review text embeddings.
  - **Performance:** Achieves ~88% R² and 0.48 RMSE.
  - **Drift Monitoring:** Includes strategies for detecting data and concept drift.

### 3. Unified Backend API (`task-3`)
A FastAPI application that integrates the RAG agent and the Rating Model into a single deployable service.
- **Endpoints:**
  - `/search`: Conversational restaurant search (RAG).
  - `/predict`: Predict user rating for a specific restaurant context.
  - `/healthz`: System health and dependency checks.
- **Deployment:** Dockerized and ready for AWS App Runner.

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized execution)
- Google Cloud API Key (for Gemini)

### Configuration
Create a `.env` file in the root directory with your API keys:
```bash
GOOGLE_API_KEY=your_api_key_here
# Optional AWS keys if deploying to SageMaker/AppRunner
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Running with Docker (Recommended)
1. Build the image:
   ```bash
   docker build -t engage-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8080:8080 --env-file .env engage-app
   ```
3. Access the API documentation at `http://localhost:8080/docs`.

### Local Development
1. Install dependencies:
   ```bash
   pip install -r task-3/backend/requirements.txt
   ```
2. Run the server:
   ```bash
   cd task-3/backend
   uvicorn app.main:app --reload
   ```

## Documentation
- **Model Performance:** See [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md) for detailed ML metrics.
- **Task 2 Report:** See `task-2/Task2_1_Report.md` for deep dive into feature engineering.
- **Values & Reflection:** See `task-1/assets/VALUES_REFLECTION.md`.

