# Video Recommender System

A production-ready collaborative filtering video recommendation system using Alternating Least Squares (ALS) algorithm with FastAPI serving and Docker support.

## Overview

This system recommends videos to users based on their interaction history (watches, likes, completions). It uses matrix factorization to discover hidden patterns in user preferences and suggest videos they're likely to enjoy.

## Project Structure

```
video-recommender/
├── data/                       # Dataset and processed files
│   ├── interactions.csv        # Raw user-video interactions
│   ├── ratings.csv             # Processed ratings (generated)
│   └── mappings.pkl            # User/video ID mappings (generated)
├── features/                   # Feature engineering
│   └── build_features.py       # Aggregates interactions into ratings
├── vrmodels/                   # ML model training and inference
│   ├── train_model.py          # Trains ALS model
│   ├── recommend.py            # Generates recommendations
│   └── als_model.pkl           # Trained model (generated)
├── serving/                    # API service
│   └── app.py                  # FastAPI application
├── evaluation/                 # Model evaluation
│   └── offline_validation.py   # Precision@K metric
├── Dockerfile                  # Docker container definition
├── serve.py                    # Entry point for running API server
├── requirements.txt            # Python dependencies
├── model.json                  # SageMaker model configuration
├── endpoint-config.json        # SageMaker endpoint configuration
├── sagemaker-trust.json        # IAM trust policy for SageMaker
└── sagemaker-s3.json           # S3 access policy for SageMaker
```

## How It Works

### 1. Feature Engineering (`features/build_features.py`)

**What it does:** Converts raw interaction data into a format suitable for training.

**Process:**
- Reads user-video interactions from `data/interactions.csv`
- Aggregates interaction values per user-video pair (e.g., watch=1, like=2, complete=3)
- Creates a rating score by summing all interaction values
- Maps user IDs and video IDs to numerical indices (e.g., "u001" → 0, "v001" → 0)
- Saves processed ratings and ID mappings

**Example:**
```
User u001 interacts with v001:
  - watch (value=1)
  - like (value=2)
  → Total rating = 3
```

**Output:**
- `data/ratings.csv` - User-video ratings with indices
- `data/mappings.pkl` - Dictionaries mapping IDs to indices

### 2. Model Training (`models/train_model.py`)

**What it does:** Trains a machine learning model to understand user preferences.

**The ALS Algorithm:**

Think of it like this: imagine you have a table where rows are users and columns are videos. Most cells are empty (users haven't watched most videos). ALS fills in the blanks by finding hidden patterns.

**How ALS works:**
1. **Matrix Decomposition**: Breaks the user-video rating matrix into two smaller matrices:
   - User factors (what each user likes)
   - Video factors (characteristics of each video)

2. **Learning Patterns**: The algorithm alternates between:
   - Fixing video factors, optimizing user factors
   - Fixing user factors, optimizing video factors
   - Repeat 20 times (iterations)

3. **Hidden Features**: Discovers 20 latent factors (features) that explain why users like certain videos
   - These might represent genres, styles, topics, etc.
   - The model learns these automatically from the data

**Key Parameters:**
- `factors=20` - Number of hidden features to discover
- `regularization=0.1` - Prevents overfitting (keeps predictions reasonable)
- `iterations=20` - How many times to refine the model

**Process:**
1. Loads `data/ratings.csv`
2. Creates a sparse matrix (5 users × 5 videos)
3. Transposes matrix to (5 videos × 5 users) for training
4. Trains the ALS model
5. Saves trained model to `models/als_model.pkl`

**The Sparse Matrix:**
```
     v001  v002  v003  v004  v005
u001   3     1     0     0     0
u002   0     0     4     0     0
u003   0     0     0     1     0
u004   0     0     0     0     3
u005   1     0     0     0     0
```

Only non-zero values are stored (memory efficient).

### 3. Making Recommendations (`models/recommend.py`)

**What it does:** Generates personalized video recommendations for a user.

**Process:**
1. Loads the trained model and ID mappings
2. Reconstructs the user-item interaction matrix
3. For a given user (e.g., "u002"):
   - Converts user ID to index (u002 → 1)
   - Extracts that user's interaction vector (row 1 from the matrix)
   - Asks the model: "What videos would this user like?"
   - Returns top N videos ranked by predicted score

**Key Detail:**
- `user_items=user_item_matrix[user_idx]` - Passes a single row (1D vector) representing which videos this user has already interacted with
- `filter_already_liked_items=True` - Excludes videos the user has already watched

**Output:**
```python
[
  ('v001', 0.91),   # Video v001, predicted score 0.91
  ('v005', 0.045),  # Video v005, predicted score 0.045
  ('v004', 0.00004),
  ...
]
```

## Data Flow

```
interactions.csv (raw data)
    ↓
[build_features.py] → Aggregates interactions
    ↓
ratings.csv + mappings.pkl
    ↓
[train_model.py] → Trains ALS model
    ↓
als_model.pkl
    ↓
[recommend.py] → Generates recommendations
    ↓
List of (video_id, score) tuples
```

## Running the System

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker installed on your machine
- Trained model files (see "Training the Model" below)

#### Build and Run
```bash
# Build the Docker image
docker buildx build \
  --platform linux/amd64 \
  --output=type=docker \
  -t video-recommender-api:latest \
  .

# Verify the architecture
docker inspect video-recommender-api:latest \
  --format='{{.Architecture}}'

# Output should be amd64

# Run the container
docker run -p 8080:8080 video-recommender-api:latest

# Or run in detached mode (background)
docker run -d -p 8080:8080 video-recommender-api:latest
```

The API will be available at `http://localhost:8080`

#### Testing the application
```bash
curl http://localhost:8080/ping

curl -X 'POST' \
  'http://localhost:8080/invocations' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "u001",
  "k": 5
}'
```
#### Stop the Application
```bash
# If running in foreground (non-detached mode)
# Press Ctrl+C in the terminal

# If running in detached mode
docker stop video-recommender-app

# To remove the stopped container
docker rm video-recommender-app

# To stop and remove in one command
docker rm -f video-recommender-app
```

#### Docker Container Details
- Base image: `python:3.10-slim`
- Exposed port: `8080`
- Entry point: `serve.py` (runs Uvicorn server)
- Includes system dependencies for scipy/implicit (gcc, g++, build-essential)
- Compatible with AWS SageMaker deployment

### Option 2: AWS Sagemaker

- Create AWS IAM user : Ex: ```ml-user```
- Refer below "AWS SageMaker Deployment" deployment section

### Option 3: Local Setup

#### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable (prevents threading issues)
export OPENBLAS_NUM_THREADS=1
```

#### Training the Model (Required Before Running API)
```bash
# 1. Build features
python features/build_features.py

# 2. Train model
python vrmodels/train_model.py

# 3. Test recommendations (optional)
python vrmodels/recommend.py
```

#### Start the API Server
```bash
# Option 1: Using serve.py
python serve.py

# Option 2: Direct uvicorn command
uvicorn serving.app:app --host 0.0.0.0 --port 8080
```

## REST API

### Base URL
```
http://localhost:8080
```

### Endpoints

#### 1. Health Check / Ping
```bash
GET /ping
```

**Example:**
```bash
curl http://localhost:8080/ping
```

**Response:**
```json
{
  "status": "ok"
}
```

#### 2. Get Recommendations (SageMaker-Compatible)
```bash
POST /invocations
Content-Type: application/json
```

**Request Body:**
```json
{
  "user_id": "u001",
  "k": 3
}
```

**Parameters:**
- `user_id` (required): User ID (e.g., "u001", "u002")
- `k` (optional): Number of recommendations to return (default: 5)

**Example:**
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u001", "k": 3}'
```

**Response:**
```json
{
  "userId": "u001",
  "recommendations": [
    {"videoId": "v005", "score": 0.9101369380950928},
    {"videoId": "v004", "score": 5.7891011238098145e-05},
    {"videoId": "v003", "score": -3.5017728805541992e-06}
  ]
}
```

**Error Responses:**
- `404`: User not found or no recommendations available
```json
{
  "detail": "User 'u999' not found"
}
```

### API Documentation
Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## AWS SageMaker Deployment

This application is designed to be compatible with AWS SageMaker for production deployment.

### Prerequisites
- AWS CLI configured with appropriate credentials
- Docker installed
- AWS ECR repository created
- SageMaker execution role with appropriate permissions

### Deployment Steps

#### 1. Build and Push Docker Image to ECR
```bash
# Create ECR Repo
aws ecr create-repository \
  --repository-name video-recommender-api \
  --region us-east-1

# Optional, but recommended
aws ecr put-image-scanning-configuration \
  --repository-name video-recommender-api \
  --image-scanning-configuration scanOnPush=true \
  --region us-east-1

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 \
| docker login \
  --username AWS \
  --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build image following command above


# Tag image
docker tag video-recommender-api:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/video-recommender-api:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/video-recommender-api:latest
```

#### 2. Create SageMaker Model
```bash
aws sagemaker create-model --cli-input-json file://model.json
```

The `model.json` file contains:
- Model name
- ECR image URI
- Execution role ARN

#### 3. Create Endpoint Configuration
```bash
aws sagemaker create-endpoint-config --cli-input-json file://endpoint-config.json
```

The `endpoint-config.json` specifies:
- Instance type (e.g., `ml.t2.medium`)
- Initial instance count
- Model name

#### 4. Create and Deploy Endpoint
Note : Billing starts from here
```bash
aws sagemaker create-endpoint \
  --endpoint-name video-recommender-endpoint \
  --endpoint-config-name video-recommender-config

# Check status
aws sagemaker describe-endpoint --endpoint-name video-recommender-endpoint --query 'EndpointStatus'
```

#### 5. Invoke the Endpoint
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name video-recommender-endpoint \
  --content-type application/json \
  --cli-binary-format raw-in-base64-out \
  --body '{"user_id": "u001", "k": 3}' \
  response.json

cat response.json
```

### SageMaker-Specific Features
- **`/ping` endpoint**: Used by SageMaker for health checks
- **`/invocations` endpoint**: Standard SageMaker inference endpoint
- **Port 8080**: Default port expected by SageMaker
- **Request/Response format**: JSON-based, compatible with SageMaker runtime

### Cleanup
```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name video-recommender-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name video-recommender-config

# Delete model
aws sagemaker delete-model --model-name video-recommender-model
```

## Key Concepts

### Collaborative Filtering
Recommends videos based on what similar users liked. If users A and B both liked videos X and Y, and user A also liked video Z, then recommend Z to user B.

### Matrix Factorization
Represents users and videos as vectors in a shared space. Videos close to a user's vector in this space are good recommendations.

### Sparse Matrix
Most users haven't watched most videos, so the matrix is mostly zeros. Sparse matrices only store non-zero values, saving memory.

### Why Transpose for Training?
The `implicit` library expects items (videos) as rows and users as columns during training. This is why we use `matrix.T` (transpose).

## Example Scenario

**User u002** has watched:
- v003 (Travel Vlog Paris) with rating 4

**Model predicts u002 might like:**
- v001 (How Python Works) - score 0.91
- v005 (Yoga for Beginners) - score 0.045

The model learned that users who enjoy travel content might also like educational or wellness content.
