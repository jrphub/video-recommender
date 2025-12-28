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

# 	userId videoId  rating
# 0   u001    v001       3
# 1   u001    v002       1
# 2   u002    v003       4
# 3   u003    v004       1
# 4   u004    v005       3
# 5   u005    v001       1

# Create ID mappings
user_map = {u: i for i, u in enumerate(ratings["userId"].unique())}
video_map = {v: i for i, v in enumerate(ratings["videoId"].unique())}

# The enumerate function is used to get both the index (i) and the value (u) of each unique user ID in the ratings["userId"] column.
print(user_map)
# {'u001': 0, 'u002': 1, 'u003': 2, 'u004': 3, 'u005': 4}

print(video_map)
# {'v001': 0, 'v002': 1, 'v003': 2, 'v004': 3, 'v005': 4}

ratings["user_idx"] = ratings["userId"].map(user_map)
ratings["video_idx"] = ratings["videoId"].map(video_map)

print(ratings)
#   userId videoId  rating  user_idx  video_idx
# 0   u001    v001       3         0          0
# 1   u001    v002       1         0          1
# 2   u002    v003       4         1          2
# 3   u003    v004       1         2          3
# 4   u004    v005       3         3          4
# 5   u005    v001       1         4          0


# Save processed data
ratings.to_csv("data/ratings.csv", index=False)

# opens a file named mappings.pkl in write mode ("wb"). This mode stands for "write binary,"
# which is necessary when dealing with binary data like Python dictionaries.
with open("data/mappings.pkl", "wb") as f:
	# serializing Data
	# The pickle module converts the Python object into a byte stream,
	# which can then be written to a file
	pickle.dump(
		{"user_map": user_map, "video_map": video_map},
		# Once the data is written to the file, the script closes it using the with block.
		# This ensures that all resources are properly released and prevents any corruption or unexpected behavior
		# if there's an error during writing.
		f
	)

print("Feature engineering completed.")
