# Quickstart

1. `cd backend && pip install -r requirements.txt`
2. `uvicorn main:app --reload --port 8000`
3. Open `frontend/index.html` directly in your browser (double-click it, or `open frontend/index.html`)
4. Try a search — e.g. "someone repairing a motorcycle while explaining the process"
5. Full API docs (auto-generated): http://127.0.0.1:8000/docs

Demo API key is `demo-key-123` (already wired into the frontend).

See `docs/README.md` for the technical write-up (VLM research, fusion
strategy, optimization plan, benchmark numbers, API contract) and
`docs/test_log.md` for verified test results.
