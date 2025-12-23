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

# userId videoId  rating
# 0   u001    v001       3
# 1   u001    v002       1
# 2   u002    v003       4
# 3   u003    v004       1
# 4   u004    v005       3
# 5   u005    v001       1

# Create ID mappings
user_map = {u: i for i, u in enumerate(ratings["userId"].unique())}
video_map = {v: i for i, v in enumerate(ratings["videoId"].unique())}

print(user_map)
# {'u001': 0, 'u002': 1, 'u003': 2, 'u004': 3, 'u005': 4}

print(video_map)
# {'v001': 0, 'v002': 1, 'v003': 2, 'v004': 3, 'v005': 4}
