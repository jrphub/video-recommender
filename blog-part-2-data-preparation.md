# Building a Video Recommendation System: Part 2 - Data Preparation and Feature Engineering

In Part 1, we introduced the concept of recommendation systems. Now, let's get our hands dirty with actual code! In this part, we'll transform raw interaction data into a format our machine learning model can understand.

## The Problem: Messy Real-World Data

Imagine you're tracking user behavior on a video platform. Your data might look like this:

```csv
userId,videoId,interactionType,interactionValue,location,timestamp
u001,v001,watch,1,USA,2025-01-01 10:10:00
u001,v001,like,2,USA,2025-01-01 10:20:00
u001,v002,watch,1,USA,2025-01-02 09:00:00
u002,v003,watch,1,UK,2025-01-03 14:30:00
u002,v003,complete,3,UK,2025-01-03 14:45:00
```

Notice the issue? User u001 has **multiple** interactions with video v001 (watch + like). Our model needs a **single rating** per user-video pair. That's where feature engineering comes in!

## What is Feature Engineering?

**Feature engineering** is the process of transforming raw data into meaningful features (inputs) that help a machine learning model learn better. Think of it as translating human actions into numbers the computer can process.

In our case, we need to:
1. Combine multiple interactions into a single "rating" score
2. Convert string IDs (like "u001", "v001") into numerical indices (0, 1, 2...)
3. Save mappings so we can translate back later

## The Code: Step by Step

Here's our complete feature engineering script (`features/build_features.py`):

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/features/build_features.py start=1
import pandas as pd
import pickle

# Load interactions
interactions = pd.read_csv("data/interactions.csv")

# Aggregate interaction strength per user-video pair
ratings = (
    interactions
    .groupby(["userId", "videoId"], as_index=False)
    .agg({"interactionValue": "sum"})
    .rename(columns={"interactionValue": "rating"})
)

print(ratings)
```

Let's break this down line by line.

### Step 1: Loading the Data

```python path=null start=null
import pandas as pd
import pickle

interactions = pd.read_csv("data/interactions.csv")
```

**What's happening?**
- `pandas` is a library for working with tabular data (think Excel, but in code)
- We read the CSV file containing all user interactions
- This loads the data into a DataFrame (a table structure)

### Step 2: Aggregating Interactions

```python path=null start=null
ratings = (
    interactions
    .groupby(["userId", "videoId"], as_index=False)
    .agg({"interactionValue": "sum"})
    .rename(columns={"interactionValue": "rating"})
)
```

**What's happening?**
- `.groupby(["userId", "videoId"])`: Group all rows by user-video combinations
- `.agg({"interactionValue": "sum"})`: For each group, sum the interaction values
- `.rename(...)`: Rename the result column to "rating"

**Example transformation:**
```
Before:
u001, v001, watch, 1
u001, v001, like, 2

After:
u001, v001, rating: 3  (1 + 2 = 3)
```

The output looks like this:

```
  userId videoId  rating
0   u001    v001       3
1   u001    v002       1
2   u002    v003       4
3   u003    v004       1
4   u004    v005       3
5   u005    v001       1
```

### Step 3: Creating ID Mappings

Machine learning models work with numbers, not strings. We need to convert "u001" to 0, "u002" to 1, etc.

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/features/build_features.py start=26
# Create ID mappings
user_map = {u: i for i, u in enumerate(ratings["userId"].unique())}
video_map = {v: i for i, v in enumerate(ratings["videoId"].unique())}
```

**What's happening?**
- `ratings["userId"].unique()`: Get all unique user IDs (e.g., ["u001", "u002", "u003", ...])
- `enumerate(...)`: Pair each ID with an index: (0, "u001"), (1, "u002"), ...
- `{u: i for i, u in ...}`: Create a dictionary mapping IDs to indices

**Result:**
```python
user_map = {'u001': 0, 'u002': 1, 'u003': 2, 'u004': 3, 'u005': 4}
video_map = {'v001': 0, 'v002': 1, 'v003': 2, 'v004': 3, 'v005': 4}
```

### Step 4: Applying the Mappings

Now we add these numerical indices to our ratings table:

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/features/build_features.py start=36
ratings["user_idx"] = ratings["userId"].map(user_map)
ratings["video_idx"] = ratings["videoId"].map(video_map)
```

**What's happening?**
- `.map(user_map)`: Replace each userId with its numerical index from our mapping
- Same for videoId

**Result:**
```
  userId videoId  rating  user_idx  video_idx
0   u001    v001       3         0          0
1   u001    v002       1         0          1
2   u002    v003       4         1          2
3   u003    v004       1         2          3
4   u004    v005       3         3          4
5   u005    v001       1         4          0
```

Perfect! Now we have both human-readable IDs and numerical indices.

### Step 5: Saving the Results

Finally, we save our processed data:

```python path=/Users/jyotiranjanpattnaik/ws_sandbox/MLApps/video-recommender/features/build_features.py start=49
# Save processed data
ratings.to_csv("data/ratings.csv", index=False)

with open("data/mappings.pkl", "wb") as f:
    pickle.dump(
        {"user_map": user_map, "video_map": video_map},
        f
    )

print("Feature engineering completed.")
```

**What's happening?**
- Save the ratings table as CSV (we'll use this for training)
- Use `pickle` to save the mapping dictionaries as a binary file
- `"wb"` means "write binary" mode

**Why pickle?** CSV is great for tables, but Python dictionaries need a special format. Pickle serializes (converts) Python objects into bytes that can be saved and loaded later.

## The Output Files

After running this script, we get two new files:

1. **`data/ratings.csv`**: Clean ratings with numerical indices
2. **`data/mappings.pkl`**: Dictionary mappings for translating between IDs and indices

These files become the input for our model training step!

## Key Concepts Explained

### Dictionary Comprehension
```python
{u: i for i, u in enumerate(items)}
```
This is a compact way to create dictionaries. It's equivalent to:
```python
result = {}
for i, u in enumerate(items):
    result[u] = i
```

### The `.map()` Function
Pandas' `.map()` function replaces values in a column using a dictionary:
```python
["u001", "u002"].map({"u001": 0, "u002": 1})
# Result: [0, 1]
```

### Why Sum Interactions?
We weight interactions differently:
- **watch** = 1 (passive engagement)
- **like** = 2 (active engagement)
- **complete** = 3 (highest engagement)

Summing these gives us a **rating** that reflects engagement strength.

## Running the Code

To run this yourself:

```bash
# Ensure you're in the project directory
cd video-recommender

# Run the feature engineering script
python features/build_features.py
```

You should see output like:
```
  userId videoId  rating  user_idx  video_idx
0   u001    v001       3         0          0
1   u001    v002       1         0          1
...
Feature engineering completed.
```

## What We Learned

✅ How to aggregate multiple interactions into single ratings  
✅ How to create numerical mappings from string IDs  
✅ How to use pandas for data transformation  
✅ How to save Python objects with pickle  
✅ Why feature engineering is crucial for machine learning

## What's Next?

Now that our data is clean and structured, we're ready to train the recommendation model! In Part 3, we'll dive into the **ALS (Alternating Least Squares)** algorithm and learn how it discovers hidden patterns in user preferences.

---

**Previous**: [← Part 1 - Introduction](blog-part-1-introduction.md)  
**Next**: [Part 3 - Training the ALS Model →](blog-part-3-model-training.md)
