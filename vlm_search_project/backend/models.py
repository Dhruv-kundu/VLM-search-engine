from pydantic import BaseModel
from typing import List, Optional, Dict


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict] = None


class SearchResult(BaseModel):
    video_id: str
    title: str
    score: float
    matched_transcript: Optional[str] = None
    matched_tags: List[str] = []
    matched_ocr: Optional[str] = None
    matched_caption: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class VideoDetail(BaseModel):
    video_id: str
    title: str
    caption: str
    transcript: str
    ocr_text: str
    objects: List[str]
    emotions: List[str]
    category: str


class SimilarRequest(BaseModel):
    top_k: int = 5
