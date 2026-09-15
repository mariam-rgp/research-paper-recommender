from fastapi import FastAPI
from pydantic import BaseModel
from src.recommendation import recommend_from_query
from src.translation import translate_abstract

app=FastAPI(
    title="Research Paper Recommender API",
    description='API for research paper Recommender API is running and abstract translation',
    version='1.0.0'
)
class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5

class TranslationRequest(BaseModel):
    text: str    

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    recommendations = recommend_from_query(
        request.query,
        request.top_k
    )

    return recommendations.to_dict(orient="records")


@app.post("/translate")
def translate(request: TranslationRequest):

    translation = translate_abstract(request.text)

    return {
        "translation": translation
    }



