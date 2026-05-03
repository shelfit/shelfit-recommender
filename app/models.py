from pydantic import BaseModel

class RecommendRequest(BaseModel):
    query: str