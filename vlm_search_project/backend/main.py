from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import SearchRequest, SearchResponse, SearchResult, VideoDetail
from search_engine import engine
from auth import verify_api_key

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Reel & Shorts Semantic Search API",
    description="B2B multimodal semantic search over short-form video content "
                 "(GGSIPU2614). Late-fusion search across visual captions, "
                 "speech transcript, OCR text, and detected objects.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to known client origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "vlm-reel-search-api", "docs": "/docs"}


@app.post("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search_videos(request: Request, req: SearchRequest, api_key: str = Depends(verify_api_key)):
    category = (req.filters or {}).get("category") if req.filters else None
    raw = engine.search(req.query, top_k=req.top_k, category=category)
    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=req.query, results=results)


@app.get("/videos/{video_id}", response_model=VideoDetail)
@limiter.limit("60/minute")
def get_video(request: Request, video_id: str, api_key: str = Depends(verify_api_key)):
    video = engine.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="video_id not found")
    return VideoDetail(**video)


@app.get("/videos/{video_id}/status")
@limiter.limit("60/minute")
def video_status(request: Request, video_id: str, api_key: str = Depends(verify_api_key)):
    video = engine.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="video_id not found")
    return {"video_id": video_id, "status": "indexed"}


@app.post("/search/similar/{video_id}", response_model=SearchResponse)
@limiter.limit("30/minute")
def similar_videos(request: Request, video_id: str, top_k: int = 5, api_key: str = Depends(verify_api_key)):
    if not engine.get_video(video_id):
        raise HTTPException(status_code=404, detail="video_id not found")
    raw = engine.similar(video_id, top_k=top_k)
    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=f"similar_to:{video_id}", results=results)
