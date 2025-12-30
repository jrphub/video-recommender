# Building a Video Recommendation System: Part 5 - Production Deployment

We've built a working recommendation engine! But it's only accessible via Python scripts. In this final part, we'll make it production-ready by creating a REST API and deploying it with Docker and AWS SageMaker.

## What We'll Build

By the end, you'll be able to send HTTP requests like:
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u001", "k": 3}'
```

And receive JSON responses:
```json
{
  "userId": "u001",
  "recommendations": [
    {"videoId": "v005", "score": 0.91},
    {"videoId": "v004", "score": 0.045}
  ]
}
```

## Part A: Building the REST API with FastAPI

FastAPI is a modern Python web framework that's fast, easy to use, and generates automatic API documentation.

### The Complete API Code

Here's our FastAPI application (`serving/app.py`):

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/serving/app.py start=1
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrmodels.recommend import recommend as get_recommendations

app = FastAPI(title="Video Recommendation API")

@app.get("/ping")
def ping():
    return {"status": "ok"}

class RecommendRequest(BaseModel):
    user_id: str
    k: int = 5

@app.post("/invocations")
def invocations(req: RecommendRequest):
    return recommend(req.user_id, req.k)

def recommend(user_id: str, k: int = 5):
    try:
        recommendations = get_recommendations(user_id, k)

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations available for user"
            )

        return {
            "userId": user_id,
            "recommendations": [
                {"videoId": video_id, "score": score}
                for video_id, score in recommendations
            ]
        }
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found"
        )
```

Let's break this down.

### Step 1: Setting Up Imports

```python path=null start=null
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrmodels.recommend import recommend as get_recommendations
```

**What's happening?**
- Import FastAPI for creating the web server
- Import HTTPException for error handling
- Import BaseModel from Pydantic for request validation
- Add the parent directory to Python's path so we can import our recommendation module
- Import our `recommend` function (rename it to avoid naming conflicts)

### Step 2: Creating the FastAPI App

```python path=null start=null
app = FastAPI(title="Video Recommendation API")
```

This creates a FastAPI application instance. The `title` appears in the auto-generated documentation.

### Step 3: Health Check Endpoint

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/serving/app.py start=13
@app.get("/ping")
def ping():
    return {"status": "ok"}
```

**Purpose:** AWS SageMaker (and many orchestration systems) use this endpoint to check if the service is healthy.

**Test it:**
```bash
curl http://localhost:8080/ping
# Response: {"status": "ok"}
```

### Step 4: Request Validation with Pydantic

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/serving/app.py start=17
class RecommendRequest(BaseModel):
    user_id: str
    k: int = 5
```

**What's Pydantic?**

Pydantic automatically:
- Validates incoming JSON against this schema
- Returns clear error messages for invalid inputs
- Converts types (e.g., string "5" → integer 5)
- Provides default values (`k` defaults to 5 if not provided)

**Example validation:**
```json
{"user_id": "u001"}          → Valid (k defaults to 5)
{"user_id": "u001", "k": 3}  → Valid
{"user_id": 123}              → Error: user_id must be string
{"k": 5}                      → Error: user_id is required
```

### Step 5: The Recommendation Endpoint

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/serving/app.py start=21
@app.post("/invocations")
def invocations(req: RecommendRequest):
    return recommend(req.user_id, req.k)
```

**Why `/invocations`?**

This is the **standard endpoint name** for AWS SageMaker. Using this convention makes deployment seamless.

**How it works:**
1. Client sends POST request with JSON body
2. FastAPI automatically parses JSON into `RecommendRequest` object
3. Validates the data (raises 422 error if invalid)
4. Calls our `recommend` function
5. Returns the result as JSON

### Step 6: The Recommendation Logic

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/serving/app.py start=25
def recommend(user_id: str, k: int = 5):
    try:
        recommendations = get_recommendations(user_id, k)

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations available for user"
            )

        return {
            "userId": user_id,
            "recommendations": [
                {"videoId": video_id, "score": score}
                for video_id, score in recommendations
            ]
        }
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found"
        )
```

**Breaking it down:**

1. **Call the recommendation engine:**
   ```python
   recommendations = get_recommendations(user_id, k)
   # Returns: [("v001", 0.91), ("v005", 0.045), ...]
   ```

2. **Handle empty results:**
   ```python
   if not recommendations:
       raise HTTPException(status_code=404, ...)
   ```

3. **Format the response:**
   ```python
   return {
       "userId": user_id,
       "recommendations": [
           {"videoId": video_id, "score": score}
           for video_id, score in recommendations
       ]
   }
   ```

   Transforms tuples into clean JSON:
   ```
   [("v001", 0.91), ("v005", 0.045)]
   →
   [{"videoId": "v001", "score": 0.91}, {"videoId": "v005", "score": 0.045}]
   ```

4. **Handle errors:**
   ```python
   except KeyError:
       raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
   ```

   If the user doesn't exist in our mappings, return a 404 error.

### The Server Entry Point

We need a script to run the API server (`serve.py`):

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/serve.py start=1
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "serving.app:app",
        host="0.0.0.0",
        port=8080
    )
```

**What's Uvicorn?**

Uvicorn is an ASGI server that runs FastAPI applications. Think of it as the "engine" that serves HTTP requests.

**Parameters:**
- `"serving.app:app"`: Import path to the FastAPI app instance
- `host="0.0.0.0"`: Accept connections from any network interface (important for containers)
- `port=8080`: Listen on port 8080 (SageMaker's standard port)

### Running the API Locally

```bash
# Option 1: Using serve.py
python serve.py

# Option 2: Using uvicorn directly
uvicorn serving.app:app --host 0.0.0.0 --port 8080
```

**Test the endpoints:**

```bash
# Health check
curl http://localhost:8080/ping

# Get recommendations
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u001", "k": 3}'
```

### Interactive API Documentation

FastAPI automatically generates documentation! Visit:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

You can test the API directly from your browser!

## Part B: Containerizing with Docker

Docker packages our application and all dependencies into a portable container that runs anywhere.

### Understanding the Dockerfile

Let's examine the project's `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for scipy/implicit
RUN apt-get update && apt-get install -y \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

ENV OPENBLAS_NUM_THREADS=1

EXPOSE 8080

CMD ["python", "serve.py"]
```

**Breaking it down:**

1. **Base image:**
   ```dockerfile
   FROM python:3.10-slim
   ```
   Start with a minimal Python 3.10 image (smaller and faster than full Python)

2. **Working directory:**
   ```dockerfile
   WORKDIR /app
   ```
   All subsequent commands run in `/app` directory

3. **System dependencies:**
   ```dockerfile
   RUN apt-get update && apt-get install -y gcc g++ build-essential
   ```
   Install compilers needed for scipy/implicit (they have C extensions)

4. **Python dependencies:**
   ```dockerfile
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   ```
   Install Python packages. `--no-cache-dir` reduces image size.

5. **Copy application code:**
   ```dockerfile
   COPY . .
   ```
   Copy all project files into the container

6. **Environment variable:**
   ```dockerfile
   ENV OPENBLAS_NUM_THREADS=1
   ```
   Prevent threading conflicts with numerical libraries

7. **Expose port:**
   ```dockerfile
   EXPOSE 8080
   ```
   Document that the container listens on port 8080

8. **Startup command:**
   ```dockerfile
   CMD ["python", "serve.py"]
   ```
   Run this command when the container starts

### Building the Docker Image

```bash
# Build for AMD64 architecture (required for AWS)
docker buildx build \
  --platform linux/amd64 \
  --output=type=docker \
  -t video-recommender-api:latest \
  .

# Verify the architecture
docker inspect video-recommender-api:latest \
  --format='{{.Architecture}}'
# Should output: amd64
```

**Why specify platform?**

If you're on an M1/M2 Mac (ARM architecture), you need to build for AMD64 (x86) because AWS EC2 instances typically use AMD64.

### Running the Container

```bash
# Run in foreground (see logs)
docker run -p 8080:8080 video-recommender-api:latest

# Run in background (detached mode)
docker run -d --name video-rec-app -p 8080:8080 video-recommender-api:latest

# Stop the container
docker stop video-rec-app

# Remove the container
docker rm video-rec-app
```

**Understanding `-p 8080:8080`:**
- Maps port 8080 on your host → port 8080 in the container
- Format: `-p HOST_PORT:CONTAINER_PORT`

## Part C: Deploying to AWS SageMaker

SageMaker is AWS's managed machine learning platform. It handles scaling, health checks, and load balancing for you.

### Prerequisites

```bash
# Install AWS CLI (if not already installed)
pip install awscli

# Configure credentials
aws configure
# Enter your Access Key ID, Secret Access Key, region (us-east-1), and output format (json)
```

### Step 1: Push Image to AWS ECR

ECR (Elastic Container Registry) is AWS's Docker registry.

```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name video-recommender-api \
  --region us-east-1

# Enable vulnerability scanning (recommended)
aws ecr put-image-scanning-configuration \
  --repository-name video-recommender-api \
  --image-scanning-configuration scanOnPush=true \
  --region us-east-1

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 \
| docker login \
  --username AWS \
  --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag the image
docker tag video-recommender-api:latest \
  <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-recommender-api:latest

# Push to ECR
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-recommender-api:latest
```

**Replace `<YOUR_ACCOUNT_ID>`** with your actual AWS account ID (12-digit number).

### Step 2: Create SageMaker Model

The project includes a `model.json` configuration file. Update it with your details:

```json
{
  "ModelName": "video-recommender-model",
  "PrimaryContainer": {
    "Image": "<YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-recommender-api:latest",
    "Mode": "SingleModel"
  },
  "ExecutionRoleArn": "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/SageMakerExecutionRole"
}
```

Create the model:
```bash
aws sagemaker create-model --cli-input-json file://model.json
```

### Step 3: Create Endpoint Configuration

Update `endpoint-config.json`:

```json
{
  "EndpointConfigName": "video-recommender-config",
  "ProductionVariants": [
    {
      "VariantName": "AllTraffic",
      "ModelName": "video-recommender-model",
      "InstanceType": "ml.t2.medium",
      "InitialInstanceCount": 1
    }
  ]
}
```

Create the configuration:
```bash
aws sagemaker create-endpoint-config --cli-input-json file://endpoint-config.json
```

**Understanding instance types:**
- `ml.t2.medium`: General-purpose, cost-effective for development/testing
- `ml.m5.large`: Better performance for production
- `ml.c5.xlarge`: Compute-optimized for heavy workloads

### Step 4: Deploy the Endpoint

```bash
# Create endpoint (⚠️ This starts billing!)
aws sagemaker create-endpoint \
  --endpoint-name video-recommender-endpoint \
  --endpoint-config-name video-recommender-config

# Check status (wait for "InService")
aws sagemaker describe-endpoint \
  --endpoint-name video-recommender-endpoint \
  --query 'EndpointStatus'
```

Deployment typically takes 5-10 minutes.

### Step 5: Invoke the Endpoint

```bash
# Make a prediction
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name video-recommender-endpoint \
  --content-type application/json \
  --cli-binary-format raw-in-base64-out \
  --body '{"user_id": "u001", "k": 3}' \
  response.json

# View the response
cat response.json
```

### Cleanup (Important!)

**SageMaker endpoints cost money even when idle!** Always clean up:

```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name video-recommender-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name video-recommender-config

# Delete model
aws sagemaker delete-model --model-name video-recommender-model

# (Optional) Delete ECR repository
aws ecr delete-repository \
  --repository-name video-recommender-api \
  --force \
  --region us-east-1
```

## Cost Considerations

**Local Docker:** Free
**AWS ECR Storage:** ~$0.10/GB/month
**AWS SageMaker:**
- ml.t2.medium: ~$0.065/hour (~$47/month if running 24/7)
- ml.m5.large: ~$0.134/hour (~$97/month)

**Cost-saving tips:**
- Use SageMaker only during development/testing
- Delete endpoints when not in use
- Consider AWS Lambda + API Gateway for low-traffic use cases

## What We Learned

✅ How to build a REST API with FastAPI
✅ Request validation with Pydantic
✅ Containerizing applications with Docker
✅ Multi-architecture Docker builds
✅ Deploying to AWS SageMaker
✅ Working with AWS ECR for container storage
✅ Production considerations (health checks, error handling)

## The Complete Workflow

```
Development:
1. python features/build_features.py
2. python vrmodels/train_model.py
3. python serve.py (test locally)

Containerization:
4. docker build -t video-recommender-api:latest .
5. docker run -p 8080:8080 video-recommender-api:latest

Production Deployment:
6. Push to AWS ECR
7. Create SageMaker model
8. Deploy endpoint
9. Invoke via AWS CLI or SDK
```

## Next Steps & Extensions

Now that you have a working system, consider:

1. **Add more features:**
   - Video metadata (genre, duration, quality)
   - User demographics (age, location, preferences)
   - Time-based patterns (time of day, day of week)

2. **Improve the model:**
   - Try different algorithms (neural collaborative filtering, LightFM)
   - Hyperparameter tuning (grid search for best factors/regularization)
   - A/B testing different recommendation strategies

3. **Scale the system:**
   - Add caching (Redis) for frequently requested users
   - Implement batch predictions
   - Set up monitoring and logging

4. **Enhance the API:**
   - Add authentication (API keys, OAuth)
   - Rate limiting to prevent abuse
   - Versioning (v1, v2 endpoints)

## Conclusion

Congratulations! You've built a complete, production-ready video recommendation system from scratch. You now understand:

- How collaborative filtering works
- The ALS algorithm and matrix factorization
- Feature engineering for machine learning
- Building REST APIs with FastAPI
- Containerization with Docker
- Cloud deployment with AWS SageMaker

This foundation applies to many recommendation scenarios: products, music, articles, social connections, and more. The concepts and code patterns you've learned are industry-standard and used by companies at scale.

Keep experimenting, keep learning, and happy coding!

---

**Previous**: [← Part 4 - Generating Recommendations](blog-part-4-recommendations.md)
**Start Over**: [Part 1 - Introduction →](blog-part-1-introduction.md)
