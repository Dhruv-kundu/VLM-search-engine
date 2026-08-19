# Verification log

Run directly in the dev sandbox against the live `uvicorn` server.

**POST /search** — query: "motorcycle repair"
→ 200 OK, top result `vid_001` (Motorcycle carburetor fix, score 0.218),
second `vid_003` (Motorcycle chain tension guide, score 0.195). Both correct
matches from a 10-video pool.

**GET /videos/vid_001**
→ 200 OK, full breakdown returned (caption, transcript, OCR, objects, emotions).

**POST /search with invalid api-key**
→ 401 Unauthorized (as designed).

**GET /videos/nonexistent**
→ 404 Not Found (as designed).

**Direct engine test** — query: "someone repairing a motorcycle while
explaining the process" (exact example from problem statement)
→ top 3: vid_001 (0.161), vid_003 (0.148), vid_007 espresso machine (0.123,
lower — correctly not an automotive match but shares "explaining process"
language, expected behavior for a text-similarity stand-in).

**Direct engine test** — query: "cooking video where chef uses air fryer"
→ top match vid_002 (Air fryer chicken wings, 0.302), vid_004 (Silent air
fryer salmon, 0.230) — vid_001 (motorcycle) scored 0.0, confirming no
false-positive cross-category matches.

**similar() test** — vid_001 (motorcycle carburetor)
→ returns vid_003 (chain tension), vid_009 (scooter repair), vid_007
(espresso descaling) — correctly clusters toward other DIY-repair-style
instructional videos.
