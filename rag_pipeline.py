#!/usr/bin/env python3
"""
End-to-end RAG pipeline for the Ice Age Trail Guidebook.

Architecture:
  Question -> Hybrid retrieval (BM25 + Voyage vector) -> RRF fusion ->
              Top-k child chunks -> Parent-document expansion ->
              Grok-4 generation with citations.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict

import voyageai
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.environ.get("CHROMA_DIR", os.path.join(_HERE, ".chroma_db"))  # built at startup if missing
CHILD_PATH      = os.path.join(_HERE, "child_chunks.jsonl")
PARENT_LOOKUP   = os.path.join(_HERE, "parent_lookup.json")
EMBEDDINGS_CACHE = os.path.join(_HERE, "embeddings_cache.pkl")

EMBED_MODEL     = "voyage-3-large"
GROK_MODEL      = "grok-4.3"
GROK_BASE_URL   = "https://api.x.ai/v1"

# Retrieval defaults (Quick-start defaults from RAG playbook)
TOP_K_VECTOR    = 30
TOP_K_BM25      = 30
TOP_K_FINAL     = 8        # how many fused candidates we send to the parent expansion
TOP_K_PARENTS   = 4        # how many parent chunks we ultimately give the LLM


# ---------------------------------------------------------------------------
# Initialize clients (singleton pattern — one per process)
# ---------------------------------------------------------------------------
_voyage = None
_chroma = None
_collection = None
_grok = None
_bm25 = None
_bm25_ids = None
_children_by_id = None
_parents_by_id = None


def get_voyage():
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])
    return _voyage


def get_collection():
    global _chroma, _collection
    if _collection is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        try:
            _collection = _chroma.get_collection("ice_age_trail_chunks")
        except Exception:
            _collection = _build_collection_from_cache(_chroma)
    return _collection


def _build_collection_from_cache(chroma_client):
    """Build the Chroma collection from cached embeddings on first startup."""
    import pickle
    print("First run: building Chroma collection from cached embeddings...")
    with open(EMBEDDINGS_CACHE, 'rb') as f:
        cache = pickle.load(f)
    children = [json.loads(l) for l in Path(CHILD_PATH).read_text().splitlines()]
    children_by_id = {c['chunk_id']: c for c in children}

    def _flatten(c):
        out = {}
        for k, v in c.items():
            if k in ('chunk_text_raw', 'chunk_text_with_header'): continue
            if v is None: continue
            if isinstance(v, (str, int, float, bool)): out[k] = v
            else: out[k] = json.dumps(v, ensure_ascii=False)
        return out

    coll = chroma_client.create_collection(
        name="ice_age_trail_chunks",
        metadata={"embedding_model": cache.get('model', 'voyage-3-large')},
    )
    ordered = [(cid, emb) for cid, emb in zip(cache['ids'], cache['embeddings']) if cid in children_by_id]
    coll.add(
        ids=[cid for cid, _ in ordered],
        embeddings=[emb for _, emb in ordered],
        documents=[children_by_id[cid]['chunk_text_with_header'] for cid, _ in ordered],
        metadatas=[_flatten(children_by_id[cid]) for cid, _ in ordered],
    )
    print(f"Built Chroma collection with {coll.count()} chunks")
    return coll


def get_grok():
    global _grok
    if _grok is None:
        _grok = OpenAI(api_key=os.environ['XAI_API_KEY'], base_url=GROK_BASE_URL)
    return _grok


def get_bm25_index():
    """Build a BM25 index over the chunk_text_raw of every child chunk."""
    global _bm25, _bm25_ids, _children_by_id
    if _bm25 is None:
        children = [json.loads(l) for l in Path(CHILD_PATH).read_text().splitlines()]
        _children_by_id = {c['chunk_id']: c for c in children}
        _bm25_ids = [c['chunk_id'] for c in children]
        # Tokenize: lowercase, split on non-word chars, drop very short tokens
        corpus = []
        for c in children:
            text = c['chunk_text_raw'].lower()
            tokens = [t for t in re.split(r'\W+', text) if len(t) > 2]
            corpus.append(tokens)
        _bm25 = BM25Okapi(corpus)
    return _bm25, _bm25_ids, _children_by_id


def get_parents():
    global _parents_by_id
    if _parents_by_id is None:
        _parents_by_id = json.loads(Path(PARENT_LOOKUP).read_text())
    return _parents_by_id


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def vector_search(query: str, top_k: int = TOP_K_VECTOR) -> List[Dict]:
    """Voyage embed the query, search Chroma, return list of {chunk_id, distance, score}."""
    vo = get_voyage()
    coll = get_collection()
    q_emb = vo.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0]
    res = coll.query(query_embeddings=[q_emb], n_results=top_k)
    out = []
    for chunk_id, dist in zip(res['ids'][0], res['distances'][0]):
        out.append({'chunk_id': chunk_id, 'distance': float(dist)})
    return out


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> List[Dict]:
    bm25, ids, _ = get_bm25_index()
    tokens = [t for t in re.split(r'\W+', query.lower()) if len(t) > 2]
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
    return [{'chunk_id': ids[i], 'bm25_score': float(s)} for i, s in indexed if s > 0]


def reciprocal_rank_fusion(rankings: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Combine multiple ranked lists into one using Reciprocal Rank Fusion.
    rankings: list of ranked lists (each item must have 'chunk_id')."""
    rrf_scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item['chunk_id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [{'chunk_id': cid, 'rrf_score': score} for cid, score in fused]


def hybrid_search(query: str, top_k_final: int = TOP_K_FINAL) -> List[Dict]:
    vector_hits = vector_search(query, TOP_K_VECTOR)
    bm25_hits   = bm25_search(query, TOP_K_BM25)
    fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
    return fused[:top_k_final]


def expand_to_parents(child_hits: List[Dict], top_k_parents: int = TOP_K_PARENTS) -> List[Dict]:
    """Group child hits by parent_chunk_id and return top parents in score order."""
    _, _, children_by_id = get_bm25_index()
    parents_by_id = get_parents()
    parent_scores = {}
    for hit in child_hits:
        c = children_by_id.get(hit['chunk_id'])
        if not c: continue
        pid = c.get('parent_chunk_id')
        if not pid: continue
        # Aggregate child scores into parent score (sum)
        parent_scores[pid] = parent_scores.get(pid, 0.0) + hit['rrf_score']
    sorted_parents = sorted(parent_scores.items(), key=lambda x: -x[1])[:top_k_parents]
    out = []
    for pid, score in sorted_parents:
        p = parents_by_id.get(pid)
        if p:
            out.append({**p, 'aggregate_score': score})
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a hiking assistant for the Ice Age National Scenic Trail in Wisconsin. You answer questions for thru-hikers using ONLY the source material provided to you below from the Ice Age Trail Alliance's official 2023 guidebook.

CRITICAL RULES:

1. Answer ONLY from the provided source material. Never speculate, never use general knowledge, never invent details. Hikers may rely on your answers for safety-critical decisions (water, shelter, hunting seasons, road crossings). Apply one of three behaviors based on what the sources contain:

   - DIRECT MATCH: if the sources directly answer the question, lead with the answer itself, in plain language. Do NOT preface it with "the guidebook does not address this question." Just give the answer with citations.

   - REFRAMED OR PARTIAL MATCH: if the sources answer the question but with different framing than the user expected (e.g., the user asks about "Polk County" but the guidebook groups Polk and Burnett together as one region), lead with what the guidebook DOES say, then briefly note the framing nuance. Example: "The guidebook groups Polk County with Burnett County as one region; the segments in this region are X, Y, Z." Do NOT lead with a refusal phrase — that implies a "no" when you actually have a real answer.

   - NO MATCH: only when the sources genuinely do not contain the answer, say "The guidebook does not specifically address this question" and briefly state what topics the sources DO cover, so the user understands what they'd need to ask differently. EXCEPTION: comparative questions (see rule 8) NEVER use this NO MATCH phrasing.

2. Cite sources for every factual claim. After each fact, include the segment name in brackets, like this: "[Bear Lake Segment]" or "[Polk & Burnett Counties]". When information comes from a sub-section, include it: "[Bear Lake Segment, AREA SERVICES]".

3. Be concise and trail-practical. Hikers reading this on a phone want quick, actionable info — distances, directions, amenities. Use bullet points or short paragraphs. Lead with the answer; details come second.

4. Preserve precision. Quote distances, GPS coordinates, road designations (CTH-V, STH-48), and proper names exactly as written in the source. Never round mileages or invent coordinates.

5. Use this glossary of guidebook conventions when expanding abbreviations or interpreting shorthand. These definitions are factual and may be cited as part of the guidebook itself:

   - **CTH** = County Trunk Highway (e.g., CTH-V is County Trunk Highway V)
   - **STH** = State Trunk Highway
   - **USH** = U.S. Highway
   - **I-** = Interstate
   - **IAT** = Ice Age Trail (the trail itself)
   - **IATA** = Ice Age Trail Alliance (the nonprofit that maintains it)
   - **NPS** = National Park Service
   - **WDNR / DNR** = Wisconsin Department of Natural Resources
   - **CR** = Connecting Route (a road-walk linking two off-road segments)
   - **TST** = Tuscobia State Trail (rail-trail shared by IAT in Barron/Washburn Co.)
   - **GDST** = Gandy Dancer State Trail (rail-trail shared by IAT in Polk/Burnett Co.)
   - **DCA** = Dispersed Camping Area (single-night camping for multi-day hikers)
   - **KMSF / KMSF-NU / KMSF-SU** = Kettle Moraine State Forest (Northern / Southern Unit)
   - **MDRA** = Mondeaux Dam Recreation Area (Taylor Co.)
   - **Trail Community** = a municipality formally recognized by IATA for hiker support
   - **ColdCache** = IATA's geocache-style program with logbooks at trailside sites
   - **Atlas Map** references the companion Ice Age Trail Atlas publication
   - **Databook** references the companion Ice Age Trail Databook

   When the user asks what an abbreviation means, this glossary is a sufficient source — you do NOT need to look it up in retrieved sources. Cite it as "[Guidebook Abbreviations]".

6. If the user's question is ambiguous (multiple possible segments, dates, or locations), ask one brief clarifying question rather than guessing.

7. If a user asks about something safety-critical (hunting, water, weather, hazards) and the sources only partially address it, be explicit about what the guidebook covers and what it doesn't, and recommend confirming with the relevant agency or the IATA before relying on the answer.

8. COMPARATIVE QUESTIONS (longest, shortest, most, fewest, biggest, smallest, hardest, easiest, all, every) require special handling. THE NO-MATCH REFUSAL FROM RULE 1 DOES NOT APPLY TO THESE QUESTIONS. The retrieved sources are a relevant subset that typically contains measurable data for several segments. You MUST:

   (a) Identify the best answer from the retrieved subset (e.g., the longest among the segments you can see).
   (b) State that answer clearly and lead with it. Example: "Of the segments in the retrieved sources, the Devil's Lake Segment is longest at 10.9 miles."
   (c) List the other retrieved segments with their measurable values for context. Example: "Other retrieved segments by length: Plover River (5.9 mi), Waterville (5.8 mi), Hartman Creek (5.5 mi)."
   (d) Add a single-sentence limitation note. Example: "The guidebook contains 100+ segments total, so for a definitive answer across the full trail, ask about a specific region (e.g., 'What's the longest segment in Sauk County?') or specific named segments."

   Under no circumstances respond to a comparative question with "The guidebook does not specifically address this question." Always provide the comparative answer from the retrieved subset, framed as such.
"""

def build_user_prompt(question: str, parent_chunks: List[Dict]) -> str:
    """Build the user prompt with retrieved sources."""
    sources_text = ""
    for i, p in enumerate(parent_chunks, start=1):
        title_parts = []
        if p.get('region'): title_parts.append(p['region'])
        if p.get('segment_name'): title_parts.append(p['segment_name'])
        title = ' > '.join(title_parts) if title_parts else (p.get('h1') or 'Document body')
        sources_text += f"\n--- SOURCE {i}: {title} ---\n\n{p['parent_text']}\n"
    return f"""Question: {question}

Source material from the Ice Age Trail Guidebook:
{sources_text}

Answer the question using only the source material above. Cite each fact with the segment or section name in brackets."""


def generate_answer(question: str, parent_chunks: List[Dict]) -> str:
    grok = get_grok()
    user_prompt = build_user_prompt(question, parent_chunks)
    resp = grok.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Top-level: ask
# ---------------------------------------------------------------------------
def ask(question: str, verbose: bool = False) -> Dict:
    """Run the full pipeline. Returns dict with answer + retrieval debug info."""
    child_hits = hybrid_search(question)
    parents = expand_to_parents(child_hits)
    answer = generate_answer(question, parents)
    return {
        'question': question,
        'answer': answer,
        'retrieved_children': child_hits if verbose else None,
        'used_parents': [{'parent_chunk_id': p['parent_chunk_id'],
                          'segment_name': p.get('segment_name'),
                          'region': p.get('region'),
                          'token_count': p.get('parent_token_count')}
                         for p in parents],
    }


if __name__ == '__main__':
    import sys
    q = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else "Where can I find water on the Bear Lake Segment?"
    result = ask(q)
    print("Q:", result['question'])
    print()
    print("A:", result['answer'])
    print()
    print("Sources used:")
    for p in result['used_parents']:
        print(f"  - {p['region']} > {p['segment_name']} ({p['token_count']} tk)")
