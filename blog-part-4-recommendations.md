# Building a Video Recommendation System: Part 4 - Generating Recommendations

We've trained our model - now let's use it! In this part, we'll write code that takes a user ID and returns personalized video recommendations. This is where everything comes together.

## The Goal

Given a user ID (like "u002"), we want to:
1. Load our trained model and mappings
2. Reconstruct the user-video interaction matrix
3. Ask the model: "What videos would this user like?"
4. Return a ranked list of recommendations

## The Complete Recommendation Script

Here's our recommendation engine (`vrmodels/recommend.py`):

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/recommend.py start=1
import pickle
import pandas as pd
from scipy.sparse import coo_matrix
import os

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load artifacts
model = pickle.load(open(os.path.join(BASE_DIR, "vrmodels", "als_model.pkl"), "rb"))
mappings = pickle.load(open(os.path.join(BASE_DIR, "data", "mappings.pkl"), "rb"))
```

Let's break this down step by step.

## Step 1: Loading Saved Artifacts

```python path=null start=null
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(BASE_DIR, "vrmodels", "als_model.pkl"), "rb"))
mappings = pickle.load(open(os.path.join(BASE_DIR, "data", "mappings.pkl"), "rb"))
```

**What's happening?**
- `BASE_DIR`: Find the project root directory (works regardless of where the script is called from)
- `pickle.load(...)`: Deserialize (load) the saved Python objects
- `"rb"` means "read binary" mode

**Why `os.path.join()`?** It creates cross-platform paths (works on Windows, Mac, Linux).

We're loading two critical files:
1. `als_model.pkl`: Our trained ALS model
2. `mappings.pkl`: Dictionaries mapping between IDs and indices

## Step 2: Preparing the Mappings

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/recommend.py start=13
user_map = mappings["user_map"]
video_map = mappings["video_map"]

reverse_video_map = {v: k for k, v in video_map.items()}
```

**Understanding the mappings:**

`video_map` goes from ID → index:
```python
{'v001': 0, 'v002': 1, 'v003': 2, 'v004': 3, 'v005': 4}
```

`reverse_video_map` goes from index → ID:
```python
{0: 'v001', 1: 'v002', 2: 'v003', 3: 'v004', 4: 'v005'}
```

**Why do we need the reverse?**
- The model returns video **indices** (0, 1, 2, ...)
- Users need video **IDs** ("v001", "v002", ...)
- The reverse map translates model output back to human-readable IDs

## Step 3: Reconstructing the User-Item Matrix

This is the trickiest part - we need to rebuild the exact matrix structure we used during training:

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/recommend.py start=20
ratings = pd.read_csv(os.path.join(BASE_DIR, "data", "ratings.csv"))

num_users = len(user_map)    # 5
num_videos = len(video_map)  # 5

# Build matrix EXACTLY like training
user_item_matrix = coo_matrix(
    (
        ratings["rating"].values,
        (ratings["user_idx"].values, ratings["video_idx"].values)
    ),
    shape=(num_users, num_videos)
).tocsr()
```

**Why rebuild the matrix?**

The model needs to know which videos each user has already interacted with. This context helps it:
- Filter out already-watched videos
- Understand user preferences based on past behavior

**What's `.tocsr()`?**

Converts the matrix to **CSR (Compressed Sparse Row)** format:
- COO (Coordinate) format: Good for building matrices
- CSR format: Optimized for fast row access (which we need!)

The result:
```
  Coords        Values
  (0, 0)        3
  (0, 1)        1
  (1, 2)        4
  (2, 3)        1
  (3, 4)        3
  (4, 0)        1
```

## Step 4: The Recommendation Function

Here's the core logic:

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/recommend.py start=58
def recommend(user_id, N=5):
    user_idx_get = user_map[user_id]
    user_idx = int(user_idx_get)
    
    ids, scores = model.recommend(
        userid=user_idx,
        user_items=user_item_matrix[user_idx],
        N=N,
        filter_already_liked_items=True
    )
    
    results = []
    for i, s in zip(ids, scores):
        video_id = reverse_video_map[i]
        score = float(s)
        results.append((video_id, score))
    
    return results
```

Let's dissect this function line by line.

### Converting User ID to Index

```python path=null start=null
user_idx_get = user_map[user_id]
user_idx = int(user_idx_get)
```

**Example:**
- Input: `user_id = "u002"`
- `user_map["u002"]` returns `1`
- Convert to integer: `user_idx = 1`

### Calling the Model's Recommend Function

```python path=null start=null
ids, scores = model.recommend(
    userid=user_idx,
    user_items=user_item_matrix[user_idx],
    N=N,
    filter_already_liked_items=True
)
```

**Breaking down the parameters:**

- **`userid=user_idx`**: Which user we're making recommendations for (e.g., user index 1)

- **`user_items=user_item_matrix[user_idx]`**: This is the KEY parameter!
  - `user_item_matrix[1]` extracts row 1 (user u002's interactions)
  - It's a **sparse vector** showing which videos this user has interacted with
  - Example: `[0, 0, 4, 0, 0]` (only interacted with video at index 2, rating 4)

- **`N=5`**: How many recommendations to return (top 5)

- **`filter_already_liked_items=True`**: Exclude videos the user has already watched
  - Why? We want to recommend NEW content, not stuff they've seen

**What does the model return?**

Two arrays:
- `ids`: Video indices `[0, 4, 3, 1]` (most recommended first)
- `scores`: Confidence scores `[0.91, 0.045, 0.00004, -3.5e-06]`

Higher scores = stronger recommendation!

### Translating Results

```python path=null start=null
results = []
for i, s in zip(ids, scores):
    video_id = reverse_video_map[i]
    score = float(s)
    results.append((video_id, score))

return results
```

**What's happening?**
- `zip(ids, scores)`: Pair each video index with its score
- `reverse_video_map[i]`: Convert index to video ID (e.g., 0 → "v001")
- `float(s)`: Convert numpy float to Python float (for JSON serialization later)
- Build a list of tuples: `[("v001", 0.91), ("v005", 0.045), ...]`

## Understanding the User-Items Vector

This is the most confusing part for beginners. Let's visualize it.

**For user u002 (index 1):**
```python
user_item_matrix[1]
```

Returns a **sparse row vector**:
```
Position:  0    1    2    3    4
Video:    v001 v002 v003 v004 v005
Rating:    0    0    4    0    0
```

This tells the model: "User 1 has only interacted with video 2, giving it a rating of 4."

The model then thinks:
- "What videos are similar to video 2?"
- "What do other users who liked video 2 also like?"
- "This user hasn't seen videos 0, 1, 3, 4 - which should I recommend?"

## Running the Recommendation Script

```bash
# Make sure you've trained the model first
python vrmodels/train_model.py

# Run recommendations for user u002
python vrmodels/recommend.py
```

**Output:**
```python
[('v001', 0.9101369380950928), 
 ('v005', 0.04477754235267639), 
 ('v004', 3.904104232788086e-05), 
 ('v002', -3.502772880554199e-06)]
```

**Interpretation:**
- **v001** has the highest score (0.91) → Strong recommendation!
- **v005** has a moderate score (0.045) → Decent recommendation
- **v004** has a tiny score → Weak recommendation
- **v002** has a negative score → Not recommended (user probably won't like it)

## The Code in the Main Block

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/recommend.py start=87
if __name__ == "__main__":
    print(recommend("u002"))
```

**What's `if __name__ == "__main__"`?**

This runs only when you execute the script directly:
```bash
python vrmodels/recommend.py  # This runs the print statement
```

But NOT when you import it:
```python
from vrmodels.recommend import recommend  # The print statement doesn't run
```

This is useful for testing!

## How the Model Generates Scores

Under the hood, the model:
1. Retrieves the user's 20-dimensional feature vector
2. Retrieves each video's 20-dimensional feature vector
3. Computes the **dot product** (multiply corresponding elements and sum)
4. Ranks videos by their dot product scores

**Example calculation:**
```
User u002 vector:  [0.5, 0.8, -0.2, 0.1, ...]  # 20 numbers
Video v001 vector: [0.6, 0.9, -0.1, 0.2, ...]  # 20 numbers

Score = (0.5 × 0.6) + (0.8 × 0.9) + (-0.2 × -0.1) + (0.1 × 0.2) + ...
      = 0.30 + 0.72 + 0.02 + 0.02 + ...
      ≈ 0.91
```

High score = user and video vectors "align" well!

## Key Concepts Recap

### Sparse Vectors
A sparse vector is mostly zeros:
```
Dense:  [0, 0, 4, 0, 0]        # Stores all 5 elements
Sparse: {2: 4}                  # Only stores non-zero positions
```

### Filtering Already-Liked Items
The model knows which videos the user has seen from `user_items`. Setting `filter_already_liked_items=True` excludes these from recommendations.

### Score Interpretation
- **Positive scores**: Model thinks user will like it
- **Higher values**: Stronger confidence
- **Negative scores**: Model thinks user won't like it

## Testing Different Users

Try recommending for different users:

```python
print(recommend("u001", N=3))  # Top 3 for user u001
print(recommend("u003", N=10)) # Top 10 for user u003
```

You'll notice different users get different recommendations based on their interaction history!

## What We Learned

✅ How to load trained models and mappings  
✅ Why we need to reconstruct the user-item matrix  
✅ How the `model.recommend()` function works  
✅ The importance of the `user_items` parameter  
✅ How to translate model outputs back to readable IDs  
✅ What recommendation scores mean  

## What's Next?

We can now generate recommendations, but they're only accessible by running a Python script. In Part 5, we'll wrap this in a **REST API** using FastAPI and deploy it with Docker and AWS SageMaker, making it accessible to web and mobile apps!

---

**Previous**: [← Part 3 - Training the Model](blog-part-3-model-training.md)  
**Next**: [Part 5 - Production Deployment →](blog-part-5-deployment.md)
