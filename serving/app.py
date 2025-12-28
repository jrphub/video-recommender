import sys
import os
from fastapi import FastAPI, HTTPException

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrmodels.recommend import recommend as get_recommendations

app = FastAPI(title="Video Recommendation API")


@app.get("/health")
def health():
	return {"status": "ok"}


@app.get("/recommend")
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
