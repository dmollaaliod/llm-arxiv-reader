#!/usr/bin/env python3
"""ArXiv Digest Analyser — Streamlit web UI."""

import asyncio
import html as html_mod
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
TOPICS_FILE   = Path("topics.txt")
OUTPUT_DIR    = Path("output")
#CLAUDE_MODEL  = "claude-opus-4-6"
CLAUDE_MODEL  = "claude-sonnet-4-6"
GEMINI_MODEL  = "gemini-2.5-flash"

PROVIDERS = {
    "Gemini": {"model": GEMINI_MODEL,  "key_env": "GOOGLE_API_KEY"},
    "Claude": {"model": CLAUDE_MODEL,  "key_env": "ANTHROPIC_API_KEY"},
}

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert research assistant specialised in analysing ArXiv paper digests.

Rules:
- ONLY return papers that are explicitly present in the digest you are given.
- Do NOT invent, hallucinate, or add papers from your own knowledge.
- Return ONLY the JSON object — no preamble, no trailing commentary.

For each relevant paper provide:
  title            – exact title as it appears in the digest
  url              – ArXiv URL; if an arXiv ID (e.g. 2401.12345) is present construct
                     https://arxiv.org/abs/<ID>; otherwise use any URL given in the digest
  summary          – 2-3 sentences summarising what the paper does
  relevance        – 2-3 sentences explaining why it is relevant to the given topic
  relevance_score  – integer 1-5 rating of relevance (5 = directly and centrally relevant)
  quality          – 2-3 sentences appraising the paper's methodology and contribution
  quality_score    – integer 1-5 rating of apparent quality (5 = excellent, rigorous work)
  venue            – conference/journal/workshop mentioned in the digest (not arXiv);
                     "Not specified" if absent

Scoring guide:
  5 – outstanding / directly central
  4 – strong / clearly relevant
  3 – moderate / tangentially relevant
  2 – weak / limited relevance or quality
  1 – minimal

JSON schema (no other text):
{
  "topic": "<topic string>",
  "papers": [
    {
      "title": "...",
      "url": "...",
      "summary": "...",
      "relevance": "...",
      "relevance_score": <1-5>,
      "quality": "...",
      "quality_score": <1-5>,
      "venue": "..."
    }
  ]
}

If no papers are relevant return: {"topic": "<topic>", "papers": []}
"""

_DIGEST_WRAPPER = """\
--- ARXIV DIGEST START ---
{digest}
--- ARXIV DIGEST END ---
"""

_TOPIC_REQUEST = """\
Identify all papers from the digest above that are relevant to this topic:

**{query}**

Return only the JSON object.
"""

# ── Topics parsing ─────────────────────────────────────────────────────────────
# Each topic is a dict: {"key": str, "query": str}
# topics.txt lines are either:
#   - plain string  →  key = query = that string
#   - JSON {"key": "...", "query": "..."}  →  key for display, query sent to LLM

def _parse_topic_line(line: str) -> "dict | None":
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
        if isinstance(data, dict) and "query" in data:
            return {"key": data.get("key", data["query"]), "query": data["query"]}
    except json.JSONDecodeError:
        pass
    return {"key": line, "query": line}


def get_topics() -> "list[dict] | str":
    if not TOPICS_FILE.exists():
        return "topics.txt not found"
    topics = [
        t for t in (_parse_topic_line(l)
                    for l in TOPICS_FILE.read_text("utf-8").splitlines())
        if t is not None
    ]
    return topics if topics else "topics.txt is empty"


# ── Digest stats ───────────────────────────────────────────────────────────────
def _parse_digest_stats(digest: str) -> dict:
    """Count new submissions, cross-listings, and revisions in an ArXiv digest."""
    lines        = digest.splitlines()
    cross_count  = sum(1 for l in lines if "(*cross-listing*)" in l)
    date_count   = sum(1 for l in lines if l.startswith("Date:"))
    repl_count   = sum(1 for l in lines if l.startswith("replaced with revised version"))
    # cross-listings also carry a Date: line, so subtract them
    new_count    = max(0, date_count - cross_count)
    return {"new": new_count, "cross_listing": cross_count, "replaced": repl_count}


def _parse_paper_types(digest: str) -> dict[str, str]:
    """Return {arxiv_id: paper_type} for every entry in the digest.

    Scans line-by-line: an `arXiv:XXXXXX` line sets the current ID (and
    whether it is a cross-listing); the next non-blank line determines the
    type ("Date:" → new/cross-listing, "replaced with…" → revised).
    """
    mapping: dict[str, str] = {}
    lines = digest.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'arXiv:(\d+\.\d+)', line)
        if m:
            arxiv_id = m.group(1)
            is_cross = "(*cross-listing*)" in line
            # Look ahead (up to 4 lines) for the type marker
            paper_type = "new"
            for j in range(i + 1, min(i + 5, len(lines))):
                nl = lines[j].strip()
                if not nl:
                    continue
                if nl.startswith("Date:"):
                    paper_type = "cross-listing" if is_cross else "new"
                    break
                if nl.startswith("replaced with revised version"):
                    paper_type = "revised"
                    break
            mapping[arxiv_id] = paper_type
        i += 1
    return mapping


def _annotate_paper_types(results: list[dict], type_map: dict[str, str]) -> None:
    """Attach paper_type to every paper in results using the pre-built type_map.

    Extracts the arXiv ID from the paper URL (strips version suffix, e.g. v2),
    looks it up in type_map, and sets paper["paper_type"]. Defaults to "new"
    when the ID is not found (e.g. non-arXiv papers).
    """
    _id_re = re.compile(r'arxiv\.org/abs/(\d+\.\d+)', re.IGNORECASE)
    for r in results:
        for p in r.get("papers", []):
            url = p.get("url", "")
            m = _id_re.search(url)
            # strip version suffix (e.g. 2603.12345v2 → 2603.12345)
            arxiv_id = re.sub(r'v\d+$', '', m.group(1)) if m else None
            p["paper_type"] = type_map.get(arxiv_id, "new") if arxiv_id else "new"


# ── Agent ──────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


MAX_RETRIES = 4          # total attempts (1 original + 3 retries)
BASE_DELAY  = 2.0       # seconds; doubles each attempt
MAX_DELAY   = 60.0      # cap for rate-limit back-off


def _backoff(attempt: int, extra: float = 0.0) -> float:
    """Exponential back-off with jitter."""
    delay = min(BASE_DELAY * math.pow(2, attempt), MAX_DELAY) + extra
    return delay + random.uniform(0, delay * 0.2)   # ±20 % jitter


async def _run_agent_claude(
    client: anthropic.AsyncAnthropic,
    topic: dict,          # {"key": ..., "query": ...}
    digest: str,
    sem: asyncio.Semaphore,
) -> dict:
    last_error = ""
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                text = ""
                async with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=16000,
                    thinking={"type": "adaptive"},
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _DIGEST_WRAPPER.format(digest=digest),
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": _TOPIC_REQUEST.format(query=topic["query"]),
                            },
                        ],
                    }],
                ) as stream:
                    async for chunk in stream.text_stream:
                        text += chunk

                result = _extract_json(text)   # raises json.JSONDecodeError on bad format
                result.setdefault("papers", [])
                result["topic_key"]   = topic["key"]
                result["topic_query"] = topic["query"]
                return result

            except anthropic.RateLimitError as exc:
                last_error = f"Rate limit: {exc}"
                if attempt < MAX_RETRIES - 1:
                    # honour Retry-After header when present, else use back-off
                    retry_after = float(
                        getattr(exc, "response", None) and
                        exc.response.headers.get("retry-after", 0) or 0
                    )
                    await asyncio.sleep(_backoff(attempt, extra=retry_after))

            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                last_error = f"API error: {exc}"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(_backoff(attempt))

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"Bad format: {exc}"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(_backoff(attempt))   # brief pause then retry

        return {
            "topic_key":   topic["key"],
            "topic_query": topic["query"],
            "papers":      [],
            "error":       f"Failed after {MAX_RETRIES} attempts — {last_error}",
        }


async def _run_agent_gemini(client, topic: dict, digest: str,
                             sem: asyncio.Semaphore) -> dict:
    from google.genai import types  # local import — only needed when Gemini is used
    last_error = ""
    prompt = (
        _DIGEST_WRAPPER.format(digest=digest) + "\n" +
        _TOPIC_REQUEST.format(query=topic["query"])
    )
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                text = ""
                stream = await client.aio.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )
                async for chunk in stream:
                    text += chunk.text or ""

                result = _extract_json(text)
                result.setdefault("papers", [])
                result["topic_key"]   = topic["key"]
                result["topic_query"] = topic["query"]
                return result

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"Bad format: {exc}"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(_backoff(attempt))

            except Exception as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES - 1:
                    exc_lower = last_error.lower()
                    extra = 30.0 if ("429" in exc_lower or "quota" in exc_lower
                                     or "exhausted" in exc_lower) else 0.0
                    await asyncio.sleep(_backoff(attempt, extra=extra))

        return {
            "topic_key":   topic["key"],
            "topic_query": topic["query"],
            "papers":      [],
            "error":       f"Failed after {MAX_RETRIES} attempts — {last_error}",
        }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _flatten(results: list[dict]) -> list[dict]:
    flat = []
    for r in results:
        for p in r.get("papers", []):
            flat.append({**p, "topic_key": r["topic_key"],
                              "topic_query": r.get("topic_query", r["topic_key"])})
    return flat


def _deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Merge papers that appear under multiple topics into one row.

    Adds:
      topics                   – ordered list of topic keys
      relevance_by_topic       – {topic_key: relevance_text}
      quality_by_topic         – {topic_key: quality_text}
      relevance_score_by_topic – {topic_key: int}
      quality_score_by_topic   – {topic_key: int}
    The top-level relevance_score / quality_score are kept as the maximum
    (used as fallback sort key); display scores are averaged per selection.
    """
    seen: dict[str, dict] = {}
    order: list[str] = []
    for p in papers:
        key = (p.get("url") or "").strip() or (p.get("title") or "").strip().lower()
        if not key:
            continue
        tk = p["topic_key"]
        rel_s  = int(p.get("relevance_score") or 0)
        qual_s = int(p.get("quality_score")   or 0)
        if key not in seen:
            seen[key] = {
                **p,
                "topics":                   [tk],
                "relevance_by_topic":       {tk: p.get("relevance", "")},
                "quality_by_topic":         {tk: p.get("quality",   "")},
                "relevance_score_by_topic": {tk: rel_s},
                "quality_score_by_topic":   {tk: qual_s},
            }
            order.append(key)
        else:
            if tk not in seen[key]["topics"]:
                seen[key]["topics"].append(tk)
            seen[key]["relevance_by_topic"][tk]       = p.get("relevance", "")
            seen[key]["quality_by_topic"][tk]         = p.get("quality",   "")
            seen[key]["relevance_score_by_topic"][tk] = rel_s
            seen[key]["quality_score_by_topic"][tk]   = qual_s
            # keep max as the canonical score (used for sorting)
            for field, val in (("relevance_score", rel_s), ("quality_score", qual_s)):
                if val > (seen[key].get(field) or 0):
                    seen[key][field] = val
    return [seen[k] for k in order]


def _save(all_papers: list[dict], results: list[dict], source_name: str,
          model: str = GEMINI_MODEL,
          digest_stats: "dict | None" = None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    matched_new   = sum(1 for p in all_papers if p.get("paper_type", "new") == "new")
    matched_cross = sum(1 for p in all_papers if p.get("paper_type") == "cross-listing")
    matched_repl  = sum(1 for p in all_papers if p.get("paper_type") == "revised")

    meta = {
        "timestamp":          datetime.now().isoformat(),
        "source":             source_name,
        "model":              model,
        "total_papers":       len(all_papers),
        "matched_new":        matched_new,
        "matched_cross":      matched_cross,
        "matched_replaced":   matched_repl,
    }
    if digest_stats:
        meta["digest_new"]          = digest_stats["new"]
        meta["digest_cross_listing"] = digest_stats.get("cross_listing", 0)
        meta["digest_replaced"]     = digest_stats["replaced"]

    jp = OUTPUT_DIR / f"arxiv_analysis_{ts}.json"
    jp.write_text(json.dumps({
        "metadata":        meta,
        "results_by_topic": results,
        "all_papers":       all_papers,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    digest_line = (
        f"**Digest:** {digest_stats['new']} new · "
        f"{digest_stats.get('cross_listing', 0)} cross-listings · "
        f"{digest_stats['replaced']} revised"
        if digest_stats else ""
    )
    mp = OUTPUT_DIR / f"arxiv_analysis_{ts}.md"
    lines = [
        "# ArXiv Digest Analysis",
        f"\n**Date:** {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**Source:** {source_name}",
        f"**Model:** `{model}`",
        *([ digest_line ] if digest_line else []),
        f"**Total papers:** {len(all_papers)} ({matched_new} new · {matched_cross} cross-listings · {matched_repl} revised)",
        "\n---\n",
    ]
    for r in results:
        key   = r.get("topic_key", "")
        query = r.get("topic_query", key)
        lines.append(f"## {key}")
        if query != key:
            lines.append(f"*Query: {query}*\n")
        papers = r.get("papers", [])
        if not papers:
            lines.append("_No relevant papers found._\n")
            continue
        for i, p in enumerate(papers, 1):
            url = p.get("url", "")
            lines += [
                f"### {i}. {p.get('title', '')}",
                f"- **URL:** [{url}]({url})",
                f"- **Venue:** {p.get('venue', 'Not specified')}",
                f"\n**Summary:** {p.get('summary', '')}",
                f"\n**Relevance:** {p.get('relevance', '')}",
                f"\n**Quality:** {p.get('quality', '')}\n",
            ]
        lines.append("---\n")
    mp.write_text("\n".join(lines), encoding="utf-8")
    return jp, mp


# ── JSON results loader ────────────────────────────────────────────────────────
def _load_json(raw: str) -> "tuple[list, list, str] | str":
    """Parse a saved results JSON.  Returns (results, all_papers, source) or an error string."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    if not isinstance(data, dict):
        return "Expected a JSON object at the top level."

    meta       = data.get("metadata", {})
    results    = data.get("results_by_topic", [])
    all_papers = data.get("all_papers", [])
    source     = meta.get("source", "loaded file")
    digest_stats = (
        {
            "new":          meta["digest_new"],
            "cross_listing": meta.get("digest_cross_listing", 0),
            "replaced":     meta["digest_replaced"],
        }
        if "digest_new" in meta else None
    )

    if not isinstance(results, list) or not isinstance(all_papers, list):
        return "JSON does not match the expected results format."

    # Back-compat: rebuild all_papers from results if the key is missing
    if not all_papers and results:
        all_papers = _flatten(results)

    # Normalise legacy field names (results saved before topic_key was added)
    for p in all_papers:
        if "topic_key" not in p:
            p["topic_key"]   = p.get("topic", "")
            p["topic_query"] = p.get("topic", "")
        # Back-compat: convert old boolean is_replacement to paper_type
        if "paper_type" not in p and "is_replacement" in p:
            p["paper_type"] = "revised" if p["is_replacement"] else "new"
    for r in results:
        if "topic_key" not in r:
            r["topic_key"]   = r.get("topic", "")
            r["topic_query"] = r.get("topic", "")

    return results, all_papers, source, digest_stats


# ── Analysis runner ────────────────────────────────────────────────────────────
def run_analysis(topics: list[dict], digest: str,
                 provider: str = "Gemini") -> list[dict]:
    progress_bar  = st.progress(0.0, text="Starting agents…")
    log_container = st.empty()
    log_lines: list[str] = []
    done = 0

    async def _all():
        nonlocal done
        if provider == "Gemini":
            from google import genai as google_genai
            client    = google_genai.Client()
            agent_fn  = _run_agent_gemini
        else:
            client    = anthropic.AsyncAnthropic()
            agent_fn  = _run_agent_claude

        sem  = asyncio.Semaphore(8)
        lock = asyncio.Lock()

        async def one(topic: dict) -> dict:
            nonlocal done
            result = await agent_fn(client, topic, digest, sem)
            async with lock:
                done += 1
                n = done
            progress_bar.progress(
                n / len(topics), text=f"({n}/{len(topics)})  ✓ {topic['key']}"
            )
            log_lines.append(f"✓ {topic['key']}")
            log_container.code("\n".join(log_lines))
            return result

        return await asyncio.gather(*[one(t) for t in topics],
                                    return_exceptions=True)

    type_map = _parse_paper_types(digest)

    raw = asyncio.run(_all())
    progress_bar.empty()
    log_container.empty()

    results = [
        r if not isinstance(r, Exception)
        else {"topic_key": topics[i]["key"], "topic_query": topics[i]["query"],
              "papers": [], "error": str(r)}
        for i, r in enumerate(raw)
    ]
    _annotate_paper_types(results, type_map)
    return results


# ── Streamlit UI ───────────────────────────────────────────────────────────────
_TABLE_CSS = """
<style>
table.arxiv-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.88em; font-family: inherit;
}
table.arxiv-table th {
    padding: 8px 12px; text-align: left;
    border-bottom: 2px solid rgba(128,128,128,0.4);
    white-space: nowrap; background: rgba(128,128,128,0.08);
}
table.arxiv-table td {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    vertical-align: top; white-space: normal; word-break: break-word;
}
table.arxiv-table tr:hover td { background: rgba(128,128,128,0.05); }
.col-topic   { width:  9%; min-width: 80px; }
.col-title   { width: 22%; min-width: 140px; }
.col-venue   { width:  9%; min-width: 80px; }
.col-score   { width:  5%; min-width: 55px; text-align: center; }
.col-url     { width:  6%; min-width: 60px; }
.col-summary { width: 49%; }
details > summary { cursor: pointer; font-weight: 600; list-style: none; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▶ "; font-size: 0.75em; opacity: 0.6; }
details[open] > summary::before { content: "▼ "; }
details[open] > summary { margin-bottom: 6px; }
.detail-section { margin-top: 8px; font-size: 0.9em; opacity: 0.85; }
.detail-label { font-weight: 600; }
.score-pip { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:2px; }
.pip-filled { background: #4CAF50; }
.pip-empty  { background: rgba(128,128,128,0.2); }
a.paper-link { color: #4e8ef7; text-decoration: none; }
a.paper-link:hover { text-decoration: underline; }
.badge-revised   { display:inline-block; padding:1px 5px; border-radius:3px;
                   font-size:0.72em; font-weight:700; background:#e67e22;
                   color:#fff; margin-left:5px; vertical-align:middle; letter-spacing:0.03em; }
.badge-crosslist { display:inline-block; padding:1px 5px; border-radius:3px;
                   font-size:0.72em; font-weight:700; background:#6c87c8;
                   color:#fff; margin-left:5px; vertical-align:middle; letter-spacing:0.03em; }
tr.row-revised   td { background: rgba(230,126,34,0.06); }
tr.row-crosslist td { background: rgba(108,135,200,0.06); }
</style>
"""

# 20 visually distinct colours (Matplotlib tab20)
_TOPIC_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _topic_color_map(all_keys: list[str]) -> dict[str, str]:
    return {
        key: _TOPIC_PALETTE[i % len(_TOPIC_PALETTE)]
        for i, key in enumerate(sorted(all_keys))
    }


def _topic_badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 7px;'
        f'border-radius:10px;font-size:0.8em;white-space:nowrap;'
        f'display:inline-block;margin:2px 0;">'
        f'{html_mod.escape(label)}</span>'
    )


def _score_pips(score: float, max_score: int = 5) -> str:
    label = f"{score:.1f}" if score != int(score) else str(int(score))
    pips = [
        f'<span class="score-pip {"pip-filled" if i <= score else "pip-empty"}"></span>'
        for i in range(1, max_score + 1)
    ]
    return f'{"".join(pips)} <small>{label}</small>'


def _avg_score(score_by_topic: dict, active_topics: list[str]) -> float:
    """Average score across active topics; fall back to all topics if none match."""
    vals = [score_by_topic[t] for t in active_topics if t in score_by_topic]
    if not vals:
        vals = list(score_by_topic.values())
    return sum(vals) / len(vals) if vals else 0.0


def _build_html_table(
    papers: list[dict],
    color_map: dict[str, str],
    sort_col: str,
    sort_asc: bool,
    selected_topics: list[str],
) -> str:
    rows = []
    for p in papers:
        title      = html_mod.escape(p.get("title", ""))
        topics     = p.get("topics", [p.get("topic_key", "")])
        topic_cell = "<br>".join(
            _topic_badge(t, color_map.get(t, "#888")) for t in topics
        )
        venue      = html_mod.escape(p.get("venue", "Not specified"))
        summary    = html_mod.escape(p.get("summary", ""))
        url        = p.get("url", "")
        url_esc    = html_mod.escape(url)

        # Use per-topic score dicts when available, else fall back to flat fields
        active = selected_topics if selected_topics else topics
        rel_score  = _avg_score(
            p.get("relevance_score_by_topic", {}), active
        ) or int(p.get("relevance_score") or 0)
        qual_score = _avg_score(
            p.get("quality_score_by_topic", {}), active
        ) or int(p.get("quality_score") or 0)

        url_cell = (
            f'<a class="paper-link" href="{url_esc}" target="_blank">🔗 Open</a>'
            if url else ""
        )

        # Build per-topic relevance/quality sections
        rel_by  = p.get("relevance_by_topic", {})
        qual_by = p.get("quality_by_topic",   {})
        # Fall back to flat fields for papers loaded from old JSON
        if not rel_by and p.get("relevance"):
            tk = topics[0] if topics else ""
            rel_by  = {tk: p["relevance"]}
            qual_by = {tk: p.get("quality", "")}

        rel_score_by = p.get("relevance_score_by_topic", {})
        qual_score_by = p.get("quality_score_by_topic", {})

        topic_sections = []
        for tk in topics:
            rel_t  = html_mod.escape(rel_by.get(tk,  ""))
            qual_t = html_mod.escape(qual_by.get(tk, ""))
            if not rel_t and not qual_t:
                continue
            badge  = _topic_badge(tk, color_map.get(tk, "#888"))
            rel_s  = rel_score_by.get(tk)
            qual_s = qual_score_by.get(tk)
            rel_pip  = f" {_score_pips(rel_s)}"  if rel_s  is not None else ""
            qual_pip = f" {_score_pips(qual_s)}" if qual_s is not None else ""
            section = (
                f'<div class="detail-section">{badge}<br>'
                + (f'<span class="detail-label">Relevance:</span>{rel_pip} {rel_t}<br><br>'
                   if rel_t else "")
                + (f'<span class="detail-label">Quality:</span>{qual_pip} {qual_t}'
                   if qual_t else "")
                + '</div>'
            )
            topic_sections.append(section)

        detail = "".join(topic_sections)
        paper_type = p.get("paper_type", "new")
        if paper_type == "revised":
            type_badge = '<span class="badge-revised">REVISED</span>'
            row_class  = ' class="row-revised"'
        elif paper_type == "cross-listing":
            type_badge = '<span class="badge-crosslist">CROSS-LIST</span>'
            row_class  = ' class="row-crosslist"'
        else:
            type_badge = ""
            row_class  = ""
        title_cell = (
            f"<details><summary>{title}{type_badge}</summary>{detail}</details>"
            if detail else f"{title}{type_badge}"
        )
        rows.append(
            f"<tr{row_class}>"
            f'<td class="col-topic">{topic_cell}</td>'
            f'<td class="col-title">{title_cell}</td>'
            f'<td class="col-venue">{venue}</td>'
            f'<td class="col-score">{_score_pips(rel_score)}</td>'
            f'<td class="col-score">{_score_pips(qual_score)}</td>'
            f'<td class="col-url">{url_cell}</td>'
            f'<td class="col-summary">{summary}</td>'
            "</tr>"
        )

    # Plain-text headers with sort indicator (no links — sorting via buttons above)
    def _th(label: str, css: str, key: str) -> str:
        indicator = (" ▲" if sort_asc else " ▼") if sort_col == key else ""
        active_style = "border-bottom:3px solid #4e8ef7;" if sort_col == key else ""
        return f'<th class="{css}" style="{active_style}">{label}{indicator}</th>'

    header = (
        "<thead><tr>"
        + _th("Topics",  "col-topic",  "topic_key")
        + _th("Title",   "col-title",  "title")
        + _th("Venue",   "col-venue",  "venue")
        + _th("Rel",     "col-score",  "relevance_score")
        + _th("Qual",    "col-score",  "quality_score")
        + '<th class="col-url">URL</th>'
        + '<th class="col-summary">Summary</th>'
        + "</tr></thead>"
    )
    return (
        f'<table class="arxiv-table">{header}'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def main() -> None:
    st.set_page_config(
        page_title="ArXiv Digest Analyser",
        page_icon="📄",
        layout="wide",
    )
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)

    # Session state defaults
    for key, val in [
        ("results",     None),
        ("all_papers",  []),
        ("source_name", ""),
        ("sort_col",    "quality_score"),
        ("sort_asc",    False),
        ("provider",    "Gemini"),
    ]:
        st.session_state.setdefault(key, val)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📄 ArXiv Digest Analyser")
        st.caption("Parallel LLM agents · one per topic")
        st.divider()

        # ── Provider selector ─────────────────────────────────────────────────
        st.subheader("Provider")
        provider = st.radio(
            "Provider",
            list(PROVIDERS.keys()),
            index=list(PROVIDERS.keys()).index(st.session_state.provider),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.provider = provider
        model_name = PROVIDERS[provider]["model"]
        key_env    = PROVIDERS[provider]["key_env"]
        import os
        if not os.environ.get(key_env):
            st.warning(f"`{key_env}` not set in environment.")
        else:
            st.caption(f"Model: `{model_name}`")
        st.divider()

        topics_or_err = get_topics()
        if isinstance(topics_or_err, str):
            st.error(topics_or_err)
            topics: list[dict] = []
        else:
            topics = topics_or_err
            st.subheader(f"Topics ({len(topics)})")
            for t in topics:
                st.markdown(f"**{t['key']}**  \n{t['query']}")

        # Filters (shown only when results exist)
        keyword = ""
        if st.session_state.results is not None:
            st.divider()
            st.subheader("Filter")
            keyword = st.text_input("Search", placeholder="keyword…")
            st.divider()
            if st.button("🔄 New Analysis", width="stretch"):
                st.session_state.results     = None
                st.session_state.all_papers  = []
                st.session_state.source_name = ""
                st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────────
    if st.session_state.results is None:

        st.header("Load Digest or Results")
        tab_upload, tab_path, tab_json = st.tabs([
            "📂 Browse & Upload",
            "⌨️  Enter File Path",
            "📊 Load JSON Results",
        ])

        digest_text  = None
        source_label = None

        with tab_upload:
            uploaded = st.file_uploader(
                "Select your ArXiv digest file",
                type=["txt", "md", "html", "csv"],
                label_visibility="collapsed",
            )
            if uploaded:
                digest_text  = uploaded.read().decode("utf-8", errors="replace")
                source_label = uploaded.name
                stats = _parse_digest_stats(digest_text)
                st.info(f"📄 **{stats['new']}** new · "
                        f"**{stats['cross_listing']}** cross-listings · "
                        f"**{stats['replaced']}** revised")

        with tab_path:
            path_input = st.text_input("File path", placeholder="/path/to/digest.txt")
            if path_input:
                p = Path(path_input.strip())
                if p.exists():
                    try:
                        digest_text  = p.read_text("utf-8", errors="replace")
                        source_label = path_input.strip()
                        stats = _parse_digest_stats(digest_text)
                        st.success(f"Loaded {p.name}  ({len(digest_text):,} chars) · "
                                   f"**{stats['new']}** new · "
                                   f"**{stats['cross_listing']}** cross-listings · "
                                   f"**{stats['replaced']}** revised")
                    except Exception as exc:
                        st.error(str(exc))
                else:
                    st.error("File not found.")

        with tab_json:
            st.caption("Load a previously saved `arxiv_analysis_*.json` file "
                       "to display its results without re-running any LLM agents.")
            json_uploaded = st.file_uploader(
                "Select a JSON results file",
                type=["json"],
                label_visibility="collapsed",
                key="json_uploader",
            )
            json_path_input = st.text_input(
                "Or enter file path",
                placeholder="/path/to/arxiv_analysis_....json",
                key="json_path",
            )

            json_raw = None
            json_name = None
            if json_uploaded:
                json_raw  = json_uploaded.read().decode("utf-8", errors="replace")
                json_name = json_uploaded.name
            elif json_path_input:
                jp = Path(json_path_input.strip())
                if jp.exists():
                    try:
                        json_raw  = jp.read_text("utf-8")
                        json_name = jp.name
                    except Exception as exc:
                        st.error(str(exc))
                else:
                    st.error("File not found.")

            if json_raw:
                outcome = _load_json(json_raw)
                if isinstance(outcome, str):
                    st.error(outcome)
                else:
                    loaded_results, loaded_papers, loaded_source, loaded_stats = outcome
                    stats_str = (
                        f" · {loaded_stats['new']} new · "
                        f"{loaded_stats.get('cross_listing', 0)} cross-listings · "
                        f"{loaded_stats['replaced']} revised"
                        if loaded_stats else ""
                    )
                    st.success(
                        f"✅ {json_name} — "
                        f"{len(loaded_papers)} paper(s) across "
                        f"{len(loaded_results)} topic(s){stats_str}"
                    )
                    if st.button("📊 Display Results", type="primary",
                                 width="stretch"):
                        st.session_state.results     = loaded_results
                        st.session_state.all_papers  = loaded_papers
                        st.session_state.source_name = loaded_source
                        st.rerun()

        # ── Analyse digest button (only relevant for the first two tabs) ───────
        st.divider()

        ready = bool(digest_text and topics)
        if not topics:
            st.warning("No topics loaded — check topics.txt.")
        if not digest_text:
            st.info("Upload or enter a digest file in the first two tabs above, "
                    "or load a saved JSON in the third tab.")

        if st.button("🔍 Analyse Digest", type="primary",
                     disabled=not ready, width="stretch"):
            with st.status("Analysing digest…", expanded=True) as status:
                results    = run_analysis(topics, digest_text, provider)
                all_papers = _flatten(results)
                try:
                    jp, _ = _save(all_papers, results, source_label,
                                  model=PROVIDERS[provider]["model"],
                                  digest_stats=_parse_digest_stats(digest_text))
                    status.update(
                        label=f"✅ Done — {len(all_papers)} paper(s) found. "
                              f"Saved to `{jp.name}`.",
                        state="complete",
                    )
                except Exception as exc:
                    status.update(label=f"Done ({exc})", state="complete")

            st.session_state.results     = results
            st.session_state.all_papers  = all_papers
            st.session_state.source_name = source_label
            st.rerun()

    else:
        # ── Results view ──────────────────────────────────────────────────────
        all_papers = st.session_state.all_papers
        results    = st.session_state.results

        # Deduplicate: papers appearing under multiple topics are merged into one row
        deduped = _deduplicate_papers(all_papers)

        # ── Inline topic filter ───────────────────────────────────────────────
        all_keys = sorted({p["topic_key"] for p in all_papers})
        selected_topics = st.multiselect(
            "Filter by topic",
            options=all_keys,
            default=[],
            placeholder="All topics",
        )

        color_map = _topic_color_map(all_keys)

        # Apply filters
        papers = deduped
        if selected_topics:
            papers = [p for p in papers
                      if any(t in selected_topics for t in p.get("topics", []))]
        if keyword:
            kw = keyword.lower()
            papers = [
                p for p in papers
                if any(kw in (p.get(f) or "").lower()
                       for f in ("title", "summary", "venue", "relevance", "quality"))
                or any(kw in t.lower() for t in p.get("topics", []))
            ]

        # ── Topic summary ─────────────────────────────────────────────────────
        with st.expander("Results by topic", expanded=False):
            st.dataframe(
                pd.DataFrame([{
                    "Key":       r.get("topic_key", ""),
                    "Query":     r.get("topic_query", ""),
                    "Papers":    len(r.get("papers", [])),
                    "New":       sum(1 for p in r.get("papers", []) if p.get("paper_type", "new") == "new"),
                    "Cross-list": sum(1 for p in r.get("papers", []) if p.get("paper_type") == "cross-listing"),
                    "Revised":   sum(1 for p in r.get("papers", []) if p.get("paper_type") == "revised"),
                    "Status":    "Error" if r.get("error") else "OK",
                    "Error":     r.get("error", ""),
                } for r in results]),
                width="stretch",
                hide_index=True,
                column_config={
                    "Key":        st.column_config.TextColumn(width="small"),
                    "Query":      st.column_config.TextColumn(width="large"),
                    "New":        st.column_config.NumberColumn(width="small"),
                    "Cross-list": st.column_config.NumberColumn(width="small"),
                    "Revised":    st.column_config.NumberColumn(width="small"),
                    "Error":      st.column_config.TextColumn(width="large"),
                },
            )

        shown_new   = sum(1 for p in papers if p.get("paper_type", "new") == "new")
        shown_cross = sum(1 for p in papers if p.get("paper_type") == "cross-listing")
        shown_repl  = sum(1 for p in papers if p.get("paper_type") == "revised")
        type_parts  = [f"{shown_new} new"]
        if shown_cross:
            type_parts.append(f"{shown_cross} cross-list")
        if shown_repl:
            type_parts.append(f"{shown_repl} revised")
        st.caption(
            f"Source: **{st.session_state.source_name}** · "
            f"Showing **{len(papers)}** papers · "
            + " · ".join(type_parts)
        )

        if not papers:
            st.info("No papers match the current filters.")
            return

        # ── Sort buttons ──────────────────────────────────────────────────────
        _SORT_COLS = [
            ("Topics", "topic_key",       True),
            ("Title",  "title",           True),
            ("Venue",  "venue",           True),
            ("Rel",    "relevance_score", False),
            ("Qual",   "quality_score",   False),
        ]
        st.caption("Sort by:")
        sort_btns = st.columns(len(_SORT_COLS))
        for (label, key, default_asc), col in zip(_SORT_COLS, sort_btns):
            active = st.session_state.sort_col == key
            indicator = (" ▲" if st.session_state.sort_asc else " ▼") if active else ""
            with col:
                if st.button(
                    f"{label}{indicator}",
                    key=f"sort_{key}",
                    width="stretch",
                    type="primary" if active else "secondary",
                ):
                    if active:
                        st.session_state.sort_asc = not st.session_state.sort_asc
                    else:
                        st.session_state.sort_col = key
                        st.session_state.sort_asc = default_asc
                    st.rerun()

        # Sort papers
        _secondary = {
            "quality_score":   "relevance_score",
            "relevance_score": "quality_score",
            "topic_key": "title", "title": "topic_key", "venue": "title",
        }
        sc  = st.session_state.sort_col
        sc2 = _secondary[sc]
        asc = st.session_state.sort_asc

        def _sort_key(p: dict):
            def val(k):
                if k == "topic_key":
                    return ", ".join(sorted(p.get("topics", [p.get("topic_key", "")])))
                if k.endswith("_score"):
                    score_by = p.get(f"{k}_by_topic", {})
                    if score_by:
                        return _avg_score(score_by, selected_topics)
                    return p.get(k) or 0
                return (p.get(k) or "").lower()
            return (val(sc), val(sc2))

        papers_sorted = sorted(papers, key=_sort_key, reverse=not asc)

        # ── HTML table ────────────────────────────────────────────────────────
        st.markdown(
            _build_html_table(papers_sorted, color_map, sc, asc, selected_topics),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
