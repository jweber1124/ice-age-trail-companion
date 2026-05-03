# Ice Age Trail Companion

A retrieval-augmented Q&A app for thru-hikers on the Ice Age National Scenic Trail. Answers come only from the official 2023 Ice Age Trail Alliance guidebook, with citations to the specific segment or section.

## Stack

- **Embedding**: Voyage AI (`voyage-3-large`, 1024 dims)
- **Retrieval**: Hybrid (BM25 + vector) with Reciprocal Rank Fusion, plus parent-document expansion
- **Vector DB**: Chroma (local, rebuilt from cached embeddings on first startup)
- **Generation**: xAI `grok-4.3` with strict source-grounding system prompt
- **Frontend**: Streamlit

## Local development

1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your Voyage and xAI API keys
4. `streamlit run app.py`

## Deployment to Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to share.streamlit.io and "New app" → connect to the repo
3. Set the main file to `app.py`
4. Under **Advanced settings → Secrets**, paste:
   ```
   VOYAGE_API_KEY = "pa-..."
   XAI_API_KEY = "xai-..."
   ```
5. Deploy

The first request will be slow (~30s) while Chroma rebuilds from `embeddings_cache.pkl`. Subsequent requests are 7–15 seconds (the bottleneck is the grok-4.3 inference call).

## Cost

- Embedding: paid for and indexed once (~$0.06 of Voyage's free 200M-token allowance)
- Per-query: ~$0.0001 Voyage (query embedding) + ~$0.01 xAI (grok-4.3 generation)

## Files

- `app.py` — Streamlit chat interface
- `rag_pipeline.py` — retrieval and generation logic
- `child_chunks.jsonl` — 870 chunks (BM25 source + retrieval results)
- `parent_lookup.json` — 212 parent chunks (LLM context)
- `embeddings_cache.pkl` — pre-computed Voyage embeddings (rebuilds Chroma on startup)
- `requirements.txt` — Python dependencies

## Disclaimer

The guidebook does not include real-time information (current trail conditions, weather, hunting season schedules, alerts). For safety-critical info, always confirm with **IceAgeTrail.org/alerts** or the relevant agency before relying on the answer.
