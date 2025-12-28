import pickle
import pandas as pd
from scipy.sparse import coo_matrix
import os

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load artifacts
model = pickle.load(open(os.path.join(BASE_DIR, "vrmodels", "als_model.pkl"), "rb"))
mappings = pickle.load(open(os.path.join(BASE_DIR, "data", "mappings.pkl"), "rb"))

user_map = mappings["user_map"]
video_map = mappings["video_map"]
# print(video_map.items())
# dict_items([('v001', 0), ('v002', 1), ('v003', 2), ('v004', 3), ('v005', 4)])
reverse_video_map = {v: k for k, v in video_map.items()}
# print(reverse_video_map)
# {0: 'v001', 1: 'v002', 2: 'v003', 3: 'v004', 4: 'v005'}
ratings = pd.read_csv(os.path.join(BASE_DIR, "data", "ratings.csv"))
# print(ratings)
#   userId videoId  rating  user_idx  video_idx
# 0   u001    v001       3         0          0
# 1   u001    v002       1         0          1
# 2   u002    v003       4         1          2
# 3   u003    v004       1         2          3
# 4   u004    v005       3         3          4
# 5   u005    v001       1         4          0

num_users = len(user_map) # 5
num_videos = len(video_map) # 5

# 🔑 Build matrix EXACTLY like training
user_item_matrix = coo_matrix(
	(
		ratings["rating"].values,
		(ratings["user_idx"].values, ratings["video_idx"].values)
	),
	shape=(num_users, num_videos)
).tocsr()
# print(user_item_matrix)
#   Coords        Values
#   (0, 0)        3
#   (0, 1)        1
#   (1, 2)        4
#   (2, 3)        1
#   (3, 4)        3
#   (4, 0)        1
item_user_matrix = user_item_matrix.T.tocsr()
# print(item_user_matrix)
#   Coords        Values
#   (0, 0)        3
#   (0, 4)        1
#   (1, 0)        1
#   (2, 1)        4
#   (3, 2)        1
#   (4, 3)        3
def recommend(user_id, N=5):
	# print(user_map)
	# {'u001': 0, 'u002': 1, 'u003': 2, 'u004': 3, 'u005': 4}
	user_idx_get = user_map[user_id]
	# print(user_idx_get) # 1
	user_idx = int(user_idx_get)
	# print(user_idx) # 1
	# print(item_user_matrix.shape)  # (5, 5)
	ids, scores = model.recommend(
		userid=user_idx,
		user_items=user_item_matrix[user_idx],  # ✅ Single user's interaction vector
		N=N,
		filter_already_liked_items=True
	)
	# print("Recommendation IDs:", ids) # [0 4 3 1 2]
	# print("Recommendation Scores:", scores) # [ 9.1016495e-01  4.4777542e-02  3.9041042e-05  3.6261976e-05 -3.4028235e+38]

	results = []

	for i, s in zip(ids, scores):
		video_id = reverse_video_map[i]
		# print(video_id)
		score = float(s)
		results.append((video_id, score))
		# print(results)

	return results


if __name__ == "__main__":
	print(recommend("u002"))
