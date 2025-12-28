import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# in terminal
# export OPENBLAS_NUM_THREADS=1

# Load data
ratings = pd.read_csv("data/ratings.csv")
print(ratings)
#   userId videoId  rating  user_idx  video_idx
# 0   u001    v001       3         0          0
# 1   u001    v002       1         0          1
# 2   u002    v003       4         1          2
# 3   u003    v004       1         2          3
# 4   u004    v005       3         3          4
# 5   u005    v001       1         4          0

num_users = ratings["user_idx"].nunique()
num_videos = ratings["video_idx"].nunique()

print(num_users) # 5
print(num_videos) # 5

# Build sparse matrix
# initializing a sparse matrix that represents interactions between users and videos.
# Each interaction (user-rating pair) is stored in the matrix,
# with the user's index as one dimension and the video's index as another dimension.
matrix = coo_matrix(
	(
		ratings["rating"],
		(ratings["user_idx"], ratings["video_idx"])
	),
	shape=(num_users, num_videos)
)

print(matrix)
# <COOrdinate sparse matrix of dtype 'int64'
#         with 6 stored elements and shape (5, 5)>
#   Coords        Values
#   (0, 0)        3
#   (0, 1)        1
#   (1, 2)        4
#   (2, 3)        1
#   (3, 4)        3
#   (4, 0)        1

"""
What is ALS?

Imagine you have a big list of movies and users who watch them. You want to recommend movies to new users based on what they've already liked.

ALS is an algorithm that helps you do this. It works by breaking down the problem into smaller parts:

Decomposing the Matrix: The first step is to break down a large matrix representing all users' ratings of movies into two smaller matrices:

A matrix where each row represents a user and each column represents a movie.
Another matrix where each row represents a user and each column represents a latent factor (a hidden feature).
Training the Model: ALS uses these two matrices to learn about how users interact with different movies. It adjusts the factors until it finds patterns that explain how users like or dislike specific movies.

Making Predictions: Once the model is trained, it can use these factors to predict how new users might feel about new movies based on what they've liked before.

Regularization
What is regularization?

Regularization is a technique used to prevent overfitting in machine learning models. Overfitting occurs when the model learns too well from the training data and performs poorly on new, unseen data.

In the context of ALS, regularization helps by adding a penalty to the size of the latent factors. This means that even if some latent factors are zero (which could mean a user doesn't have any interest in certain types of movies), they won't be considered part of the model's output.

The idea is to encourage the model to focus on relevant features rather than fitting to noise or irrelevant patterns. Regularization helps keep the model simpler and more robust, making it less likely to overfit to the training data.

Why Use ALS?
Handling Sparse Data: ALS is particularly useful for handling sparse datasets, where many elements are zero (indicating that a user hasn't interacted with certain movies). This makes it suitable for large-scale recommendation systems.

Flexibility: ALS can handle various types of rating data (e.g., binary ratings, continuous ratings), making it versatile for different recommendation scenarios.

Interpretability: The latent factors learned by ALS can provide insights into what features a user likes or dislikes, which can be helpful for understanding how users interact with movies.
"""
# Train ALS model
model = AlternatingLeastSquares(
	factors=20,
	regularization=0.1,
	iterations=20
)

print(model)
# T is for transpose of matrix
# where each row represents a movie and each column represents a user.
model.fit(matrix.T)

# Save model
with open("models/als_model.pkl", "wb") as f:
	pickle.dump(model, f)

print("Model training completed.")
