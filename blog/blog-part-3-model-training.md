# Building a Video Recommendation System: Part 3 - Training the ALS Model

Welcome back! In Part 2, we prepared our data. Now comes the exciting part: teaching a machine learning model to predict what users will like. We'll use the **ALS (Alternating Least Squares)** algorithm - don't worry, we'll explain everything in simple terms!

## What Problem Are We Solving?

Imagine a giant table where:
- Each **row** represents a user
- Each **column** represents a video
- Each **cell** contains a rating (or is empty if the user hasn't watched that video)

```
        v001  v002  v003  v004  v005
u001      3     1     ?     ?     ?
u002      ?     ?     4     ?     ?
u003      ?     ?     ?     1     ?
u004      ?     ?     ?     ?     3
u005      1     ?     ?     ?     ?
```

**Our goal:** Fill in those question marks! If we can predict missing ratings, we can recommend videos users haven't seen yet.

## What is ALS (Alternating Least Squares)?

ALS is a clever algorithm that **factorizes** this big table into two smaller tables:

1. **User features**: What each user likes (20 hidden characteristics)
2. **Video features**: What each video offers (20 hidden characteristics)

Think of these hidden characteristics like personality traits:
- Maybe feature #1 represents "educational content"
- Maybe feature #2 represents "entertainment value"
- Maybe feature #3 represents "production quality"

The model learns these automatically from the data!

### The Matrix Factorization Concept

```
Rating Matrix (5×5)  =  User Matrix (5×20)  ×  Video Matrix (20×5)
  [partially filled]       [dense/full]          [dense/full]
```

By multiplying user features × video features, we can predict ANY user-video rating!

### Why "Alternating"?

ALS alternates between:
1. **Fix videos, optimize users**: "Given what we know about videos, what must users like?"
2. **Fix users, optimize videos**: "Given what we know about users, what must videos be like?"
3. Repeat 20 times until the predictions are accurate

It's like solving a puzzle by alternating between two perspectives!

## The Code: Step by Step

Here's our model training script (`vrmodels/train_model.py`):

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/train_model.py start=1
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# Load data
ratings = pd.read_csv("../data/ratings.csv")
```

### Step 1: Loading Dependencies

```python path=null start=null
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
```

**What are these?**
- `coo_matrix`: Creates a **sparse matrix** (efficient storage for mostly-empty data)
- `AlternatingLeastSquares`: The ALS algorithm implementation from the `implicit` library

### Step 2: Understanding the Data

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/train_model.py start=10
ratings = pd.read_csv("data/ratings.csv")
print(ratings)

num_users = ratings["user_idx"].nunique()
num_videos = ratings["video_idx"].nunique()

print(num_users)   # 5
print(num_videos)  # 5
```

Our ratings table looks like:
```
  userId videoId  rating  user_idx  video_idx
0   u001    v001       3         0          0
1   u001    v002       1         0          1
2   u002    v003       4         1          2
3   u003    v004       1         2          3
4   u004    v005       3         3          4
5   u005    v001       1         4          0
```

We have 5 unique users and 5 unique videos.

### Step 3: Building the Sparse Matrix

Here's where the magic happens:

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/train_model.py start=26
# Build sparse matrix
matrix = coo_matrix(
    (
        ratings["rating"],
        (ratings["user_idx"], ratings["video_idx"])
    ),
    shape=(num_users, num_videos)
)

print(matrix)
```

**What's a sparse matrix?**

In our 5×5 table, only 6 cells have values. The other 19 cells are empty (zero). A **sparse matrix** only stores the non-zero values to save memory.

**Output:**
```
<COOrdinate sparse matrix of dtype 'int64'
        with 6 stored elements and shape (5, 5)>
  Coords        Values
  (0, 0)        3
  (0, 1)        1
  (1, 2)        4
  (2, 3)        1
  (3, 4)        3
  (4, 0)        1
```

**Breaking down the syntax:**
```python
coo_matrix(
    (values, (row_indices, col_indices)),
    shape=(rows, cols)
)
```

- `values`: The ratings [3, 1, 4, 1, 3, 1]
- `row_indices`: User indices [0, 0, 1, 2, 3, 4]
- `col_indices`: Video indices [0, 1, 2, 3, 4, 0]
- `shape`: Overall dimensions (5 users × 5 videos)

This efficiently represents our partially-filled rating table!

### Step 4: Configuring the ALS Model

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/train_model.py start=81
# Train ALS model
model = AlternatingLeastSquares(
    factors=20,
    regularization=0.1,
    iterations=20
)
```

**Understanding the parameters:**

- **`factors=20`**: The number of hidden features to discover
  - Higher = more expressive but slower and risk overfitting
  - Lower = faster but less accurate
  - 20 is a good balance for most systems

- **`regularization=0.1`**: Prevents overfitting
  - Adds a penalty for overly complex patterns
  - Keeps predictions reasonable even for sparse data
  - Think of it as "don't overinterpret limited data"

- **`iterations=20`**: How many times to alternate between optimizing users and videos
  - More iterations = better accuracy (up to a point)
  - Too many = overfitting
  - 20 is typically sufficient for convergence

### Step 5: Training the Model

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/train_model.py start=88
# T is for transpose of matrix
# where each row represents a video and each column represents a user.
model.fit(matrix.T)
```

**Wait, why `.T` (transpose)?**

The `implicit` library expects the matrix in a specific orientation:
- **Rows = items (videos)**
- **Columns = users**

But we built it as:
- **Rows = users**
- **Columns = videos**

So we transpose (flip rows and columns) using `.T`:

```
Original (5 users × 5 videos):
        v001  v002  v003  v004  v005
u001      3     1     0     0     0
u002      0     0     4     0     0
...

Transposed (5 videos × 5 users):
        u001  u002  u003  u004  u005
v001      3     0     0     0     1
v002      1     0     0     0     0
v003      0     4     0     0     0
...
```

**What happens during training?**

The model:
1. Initializes random user and video feature vectors
2. Alternates between fixing one and optimizing the other
3. Minimizes prediction error across all known ratings
4. Converges after 20 iterations

This typically takes a few seconds on small datasets!

### Step 6: Saving the Trained Model

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/vrmodels/train_model.py start=93
# Save model
with open("models/als_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training completed.")
```

We save the trained model using `pickle` so we can load it later for making predictions without retraining.

## Key Concepts Deep Dive

### What is Regularization?

Imagine you're learning to ride a bike by watching 3 people. Without regularization, you might conclude:
- "Everyone who rides bikes wears red shoes!"

Regularization says:
- "Maybe that's just coincidence. Don't rely too heavily on patterns from limited data."

In math terms, it adds a penalty: "Keep feature values small unless really necessary."

### Why Sparse Matrices?

Consider a real system with:
- 1 million users
- 100,000 videos
- Each user watches 50 videos on average

A dense matrix would require: 1M × 100K = 100 billion cells!
A sparse matrix stores only: 1M × 50 = 50 million values (2000× smaller!)

### The 20 Hidden Factors

The model learns 20 numbers for each user and each video. Example:

**User u001's vector (simplified):**
```
[0.8, 0.2, 0.9, -0.1, 0.5, ...]  # 20 numbers
 ↑    ↑    ↑     ↑     ↑
 likes educational  dislikes prefers
 tech  comedy       vlogs    short
```

**Video v001's vector (simplified):**
```
[0.9, 0.1, 0.8, -0.2, 0.4, ...]  # 20 numbers
 ↑    ↑    ↑     ↑     ↑
 very  a bit of  pretty  not a   medium
 tech  comedy    edu     vlog    length
```

**Predicted rating = dot product:**
```
0.8×0.9 + 0.2×0.1 + 0.9×0.8 + ... ≈ 3.2
```

This predicted rating tells us how much u001 would like v001!

## Running the Training

Before training, make sure you've run the feature engineering step:

```bash
# Step 1: Build features (if you haven't already)
python features/build_features.py

# Step 2: Train the model
export OPENBLAS_NUM_THREADS=1  # Prevents threading issues
python vrmodels/train_model.py
```

**Note:** The `OPENBLAS_NUM_THREADS=1` environment variable prevents conflicts with numerical libraries.

You should see output like:
```
  userId videoId  rating  user_idx  video_idx
0   u001    v001       3         0          0
...
5
5
<COOrdinate sparse matrix...>
Model training completed.
```

## What Just Happened?

After training, our model has learned:
- A 5×20 matrix of user features
- A 20×5 matrix of video features
- How to multiply them to predict any user-video rating

This trained model is saved as `models/als_model.pkl` and is ready to make recommendations!

## Evaluation: How Good Is Our Model?

In production systems, you'd evaluate the model using:
- **Precision@K**: "Of the top K recommendations, how many did the user actually like?"
- **Recall@K**: "Of all videos the user liked, how many were in our top K?"
- **NDCG**: Normalized Discounted Cumulative Gain (rewards ranking quality)

Our project includes a simple evaluation function in `evaluation/offline_validation.py`:

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/evaluation/offline_validation.py start=1
def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / k
```

This calculates: "What fraction of our top K recommendations were actually relevant?"

## What We Learned

✅ What matrix factorization means and why it's powerful
✅ How ALS alternates between user and video optimization
✅ Why sparse matrices are essential for recommendation systems
✅ The role of regularization in preventing overfitting
✅ How to train a collaborative filtering model with real code

## What's Next?

Now that our model is trained, it's time to use it! In Part 4, we'll write code to generate personalized recommendations for any user and understand how predictions work under the hood.

---

**Previous**: [← Part 2 - Data Preparation](blog-part-2-data-preparation.md)
**Next**: [Part 4 - Generating Recommendations →](blog-part-4-recommendations.md)
