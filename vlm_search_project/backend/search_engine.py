"""
Search engine implementing the LATE FUSION strategy from Step 2.

In the real pipeline, `caption` comes from BLIP-2/Florence-2, `transcript`
from Whisper, `ocr_text` from EasyOCR, and `objects` from YOLOv11 — each run
once per video during ingestion and cached. This module then fuses those
fields at SEARCH time by scoring the query against each field separately
and combining the scores (see FIELD_WEIGHTS below).

This demo replaces the CLIP/BLIP-2 embedding step with TF-IDF text
similarity so it runs without a GPU or model downloads. To go from this
prototype to the real pipeline:
  1. Replace `_vectorize_field` with a call to the appropriate VLM/ASR model
     (see docs/README.md Step 1-3) and cache the resulting embeddings.
  2. Replace `cosine_similarity` on TF-IDF vectors with cosine similarity on
     the real embeddings (same interface, same fusion logic below).
  3. Everything downstream (API, frontend, explainability) is unchanged.
"""
import json
import os
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_videos.json")

# Late-fusion weights per field — tune these against your Recall@K benchmark
# (see docs/README.md Step 4). Caption/transcript carry semantic meaning,
# OCR and objects are exact-match signals so they get less weight here.
FIELD_WEIGHTS = {
    "caption": 0.35,
    "transcript": 0.30,
    "ocr_text": 0.10,
    "objects": 0.25,
}


class FusionSearchEngine:
    def __init__(self, data_path: str = DATA_PATH):
        with open(data_path, "r") as f:
            self.videos: List[Dict] = json.load(f)

        self.vectorizers = {}
        self.field_matrices = {}

        for field in FIELD_WEIGHTS:
            texts = [self._field_text(v, field) for v in self.videos]
            vectorizer = TfidfVectorizer(stop_words="english")
            # guard against an all-empty field (e.g. no video has ocr_text)
            if any(t.strip() for t in texts):
                matrix = vectorizer.fit_transform(texts)
            else:
                matrix = None
            self.vectorizers[field] = vectorizer
            self.field_matrices[field] = matrix

    @staticmethod
    def _field_text(video: Dict, field: str) -> str:
        val = video.get(field, "")
        if isinstance(val, list):
            return " ".join(val)
        return val or ""

    def _score_field(self, query: str, field: str):
        matrix = self.field_matrices[field]
        if matrix is None:
            return [0.0] * len(self.videos)
        vectorizer = self.vectorizers[field]
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, matrix)[0]
        return sims

    def search(self, query: str, top_k: int = 10, category: Optional[str] = None) -> List[Dict]:
        field_scores = {field: self._score_field(query, field) for field in FIELD_WEIGHTS}

        fused = []
        for i, video in enumerate(self.videos):
            if category and video.get("category") != category:
                continue
            score = sum(FIELD_WEIGHTS[f] * field_scores[f][i] for f in FIELD_WEIGHTS)
            fused.append((i, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in fused[:top_k]:
            video = self.videos[i]
            results.append({
                "video_id": video["video_id"],
                "title": video["title"],
                "score": round(float(score), 4),
                "matched_transcript": video["transcript"][:140] if video["transcript"] else None,
                "matched_tags": video["objects"],
                "matched_ocr": video["ocr_text"] or None,
                "matched_caption": video["caption"],
            })
        return results

    def get_video(self, video_id: str) -> Optional[Dict]:
        for v in self.videos:
            if v["video_id"] == video_id:
                return v
        return None

    def similar(self, video_id: str, top_k: int = 5) -> List[Dict]:
        """Use the video's own caption+transcript as the query — a simple
        stand-in for comparing stored embeddings directly."""
        video = self.get_video(video_id)
        if not video:
            return []
        pseudo_query = f"{video['caption']} {video['transcript']}"
        results = self.search(pseudo_query, top_k=top_k + 1)
        return [r for r in results if r["video_id"] != video_id][:top_k]


engine = FusionSearchEngine()
