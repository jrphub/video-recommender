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
