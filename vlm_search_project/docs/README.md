# Reel & Shorts Semantic Search — GGSIPU2614
### Scope covered: VLM Research & Optimization · UX/Frontend · B2B APIs

This is a **working prototype** of the search, API, and frontend layers, running
end-to-end against a small demo dataset (`backend/data/sample_videos.json`,
10 videos). It uses TF-IDF text similarity in place of real CLIP/BLIP-2/Whisper
inference so it runs without a GPU or model downloads — the **architecture,
fusion logic, API contract, and frontend are all real** and match the design
decisions below. Swapping in real models is a drop-in change (see "Going to
production" at the bottom).

---

## 1. VLM Research & Optimization

### Models selected (per problem statement's modality list)
| Modality | Model | Role |
|---|---|---|
| Visual scene | CLIP (ViT-B/32) | Fast image↔text embedding for visual similarity |
| Dense captioning | BLIP-2 / Florence-2 | Per-video scene description |
| Complex/relational reasoning | LLaVA / Gemini API | Used sparingly, only for re-ranking compositional queries (e.g. "repairing X while explaining Y") — too slow to run on every frame |
| Speech | Whisper (base/small) | Transcript extraction |
| On-screen text | EasyOCR | Caption/subtitle/label text |
| Objects/actions | YOLOv11 (nano) | Object and activity tags |

### Fusion strategy: late fusion
Each modality is embedded independently, and similarity scores are combined
at query time rather than concatenating raw embeddings ("early fusion").
Chosen because it's easier to debug, each field's contribution to a match is
directly explainable in the UI, and it degrades gracefully when a field is
empty (e.g. a silent video with no transcript).

```
FIELD_WEIGHTS = {
    "caption": 0.35,     # BLIP-2/Florence-2 scene description
    "transcript": 0.30,  # Whisper
    "ocr_text": 0.10,    # EasyOCR
    "objects": 0.25,     # YOLO tags
}
final_score = sum(weight * cosine_similarity(query, field) for field, weight in FIELD_WEIGHTS)
```

Weights are a starting point tuned by inspection on the demo set — in
production, sweep these against a real Recall@K benchmark (Step 4 below)
rather than hand-picking them.

See `backend/search_engine.py` for the full implementation — the fusion
logic is model-agnostic, so it's unchanged whether the field vectors come
from TF-IDF (this demo) or real CLIP/BLIP-2 embeddings (production).

### Optimization plan (for the real pipeline)
1. **Keyframe sampling** — scene-change detection (frame-diff threshold via
   OpenCV) instead of fixed-interval sampling. Typically cuts frames-per-video
   by 60-70% with minimal recall loss.
2. **Batched inference** — batch CLIP calls across frames instead of one
   image at a time.
3. **Quantization** — export CLIP/BLIP-2 to ONNX with INT8/FP16 precision via
   `optimum[onnxruntime]`. Typically 2-4x latency reduction.
4. **Model size** — YOLO nano, Whisper base/small, BLIP-2 run once per
   keyframe (not every frame) unless benchmarking shows accuracy requires more.

### Benchmark results (demo dataset, 5 test queries, 10 videos)
| Metric | Score |
|---|---|
| Recall@1 | 100% |
| Recall@3 | 100% |
| Recall@5 | 100% |

This is a **sanity check on a small, clean demo set** — not a substitute for
benchmarking on MSR-VTT/YouCook2 with hundreds of videos and near-duplicate
content, which is necessary to get a meaningful, defensible number for the
project report. Run the same `recall_at_k` pattern (see
`backend/search_engine.py` usage in the dev notes) against a real benchmark
subset once the real embedding pipeline is in place.

---

## 2. B2B API

FastAPI service (`backend/main.py`), OpenAPI docs auto-generated at `/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/search` | POST | Natural language query → ranked, explainable results |
| `/videos/{video_id}` | GET | Full multimodal breakdown for one video |
| `/videos/{video_id}/status` | GET | Ingestion/indexing status |
| `/search/similar/{video_id}` | POST | Similarity-based recommendations |

- **Auth**: API key via `api-key` header (`backend/auth.py`)
- **Rate limiting**: 30-60 req/min per client (slowapi)
- **CORS**: open for demo, restrict `allow_origins` to known client domains in production

Tested and verified working: search returns correctly ranked results,
invalid key → 401, missing video → 404 (see test transcript in
`docs/test_log.md` — verified by directly running the server and querying
each endpoint).

---

## 3. UX / Frontend

Single-page app (`frontend/index.html`, vanilla JS — no build step needed).

- **Search bar** — natural language input, example-query chips for discoverability
- **Explainable results grid** — every result card shows *why* it matched:
  quoted transcript snippet, matched OCR text, detected object tags, and a
  signal-strength meter for the fusion score
- **Video Understanding Dashboard** — click any result to see the full
  per-video breakdown (caption, transcript, OCR, objects, emotion signals)
- **Similarity recommendations** — "Find similar videos" from the dashboard

Design direction: dark, technical "signal extraction" aesthetic (filmstrip
motif, level-meter score visualization, monospace for data/tags vs. sans for
prose) — deliberately distinct from generic AI-tool defaults, reflecting the
subject (extracting signals from raw video).

---

## 4. Running it

```bash
# backend
cd backend
pip install fastapi uvicorn scikit-learn slowapi pydantic
uvicorn main:app --reload --port 8000

# frontend
# just open frontend/index.html in a browser (it calls http://127.0.0.1:8000)
```

Demo API key: `demo-key-123`

---

## 5. Going to production (swapping in real models)

Everything in this prototype is structured so only `search_engine.py`'s
`_vectorize_field`/`_score_field` methods need to change:

1. **Ingestion**: build the pipeline from Steps 1-3 of the project plan —
   FFmpeg keyframe extraction → CLIP/BLIP-2/Whisper/YOLO/EasyOCR per video →
   cache embeddings (e.g. in Postgres + FAISS/Milvus instead of the JSON file).
2. **Search**: replace TF-IDF cosine similarity with real embedding cosine
   similarity against the vector DB. The `FIELD_WEIGHTS` late-fusion logic
   stays exactly the same.
3. **API and frontend**: unchanged — they only depend on the `search()` /
   `get_video()` / `similar()` interface, not on how embeddings are produced.
