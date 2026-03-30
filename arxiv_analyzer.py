#!/usr/bin/env python3
"""
ArXiv Digest Analyzer
Spawns parallel LLM agents (one per topic) to extract relevant papers from an ArXiv digest.
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from dotenv import load_dotenv

load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────
TOPICS_FILE = Path("topics.txt")
OUTPUT_DIR = Path("output")
MODEL = "claude-opus-4-6"

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert research assistant specialised in analysing ArXiv paper digests.

Rules:
- ONLY return papers that are explicitly present in the digest provided to you.
- Do NOT invent, hallucinate, or add papers from your own knowledge.
- Return ONLY the JSON object — no preamble, no trailing commentary.

For each relevant paper provide:
  title    – exact title as it appears in the digest
  url      – ArXiv URL; if an arXiv ID (e.g. 2401.12345) is present construct
             https://arxiv.org/abs/<ID>; otherwise use any URL given in the digest
  summary  – 2-3 sentences summarising what the paper does
  relevance– 2-3 sentences explaining why it is relevant to the given topic
  quality  – 2-3 sentences appraising the paper's methodology and contribution
  venue    – conference/journal/workshop mentioned in the digest (not arXiv);
             "Not specified" if absent

JSON schema (no other text):
{
  "topic": "<topic string>",
  "papers": [
    {
      "title": "...",
      "url": "...",
      "summary": "...",
      "relevance": "...",
      "quality": "...",
      "venue": "..."
    }
  ]
}

If no papers are relevant return: {"topic": "<topic>", "papers": []}
"""

DIGEST_BLOCK_TEMPLATE = """\
--- ARXIV DIGEST START ---
{digest}
--- ARXIV DIGEST END ---
"""

TOPIC_BLOCK_TEMPLATE = """\
Identify all papers from the digest above that are relevant to this topic:

**{topic}**

Return only the JSON object.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
console = Console()


def read_topics() -> list[str]:
    if not TOPICS_FILE.exists():
        console.print(f"[red]Error: {TOPICS_FILE} not found.[/red]")
        sys.exit(1)
    topics = [
        line.strip()
        for line in TOPICS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not topics:
        console.print("[red]Error: topics.txt is empty.[/red]")
        sys.exit(1)
    return topics


def read_digest(path: str) -> str:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error: digest file '{path}' not found.[/red]")
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def extract_json(text: str) -> dict:
    """Extract the first top-level JSON object from text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


# ── Agent ─────────────────────────────────────────────────────────────────────
async def run_topic_agent(
    client: anthropic.AsyncAnthropic,
    topic: str,
    digest: str,
    semaphore: asyncio.Semaphore,
    on_done,
) -> dict:
    """Run one LLM agent for a single topic."""
    async with semaphore:
        try:
            full_text = ""
            async with client.messages.stream(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            # Cache the digest — shared across all agents
                            {
                                "type": "text",
                                "text": DIGEST_BLOCK_TEMPLATE.format(digest=digest),
                                "cache_control": {"type": "ephemeral"},
                            },
                            # Topic-specific part — not cached
                            {
                                "type": "text",
                                "text": TOPIC_BLOCK_TEMPLATE.format(topic=topic),
                            },
                        ],
                    }
                ],
            ) as stream:
                async for chunk in stream.text_stream:
                    full_text += chunk

            result = extract_json(full_text)
            result.setdefault("papers", [])
            result["topic"] = topic
        except Exception as exc:
            result = {"topic": topic, "papers": [], "error": str(exc)}

        on_done()
        return result


async def run_all_agents(
    topics: list[str], digest: str, max_concurrent: int = 8
) -> list[dict]:
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            f"[cyan]Analysing {len(topics)} topic(s)…", total=len(topics)
        )

        def on_done():
            progress.advance(task_id)

        coros = [
            run_topic_agent(client, topic, digest, semaphore, on_done)
            for topic in topics
        ]
        raw = await asyncio.gather(*coros, return_exceptions=True)

    results = []
    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            results.append({"topic": topics[i], "papers": [], "error": str(r)})
        else:
            results.append(r)
    return results


# ── Presentation ──────────────────────────────────────────────────────────────
def flatten_papers(results: list[dict]) -> list[dict]:
    flat = []
    for r in results:
        for p in r.get("papers", []):
            flat.append({**p, "topic": r["topic"]})
    return flat


def make_table(papers: list[dict], title: str) -> Table:
    tbl = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        highlight=True,
        title_style="bold cyan",
    )
    tbl.add_column("#", style="dim", width=3, justify="right", no_wrap=True)
    tbl.add_column("Topic", style="cyan", max_width=22, overflow="fold")
    tbl.add_column("Title", style="bold white", max_width=36, overflow="fold")
    tbl.add_column("Venue", style="green", max_width=18, overflow="fold")
    tbl.add_column("URL", style="blue", max_width=38, overflow="fold")
    tbl.add_column("Summary", max_width=42, overflow="fold")
    tbl.add_column("Relevance", max_width=38, overflow="fold")
    tbl.add_column("Quality", max_width=38, overflow="fold")

    for i, p in enumerate(papers, 1):
        tbl.add_row(
            str(i),
            p.get("topic", ""),
            p.get("title", ""),
            p.get("venue", "Not specified"),
            p.get("url", ""),
            p.get("summary", ""),
            p.get("relevance", ""),
            p.get("quality", ""),
        )
    return tbl


def show_topic_summary(results: list[dict]) -> None:
    tbl = Table(
        title="Results by Topic",
        box=box.SIMPLE_HEAD,
        show_header=True,
        title_style="bold cyan",
    )
    tbl.add_column("Topic", style="cyan")
    tbl.add_column("Papers", justify="center", style="bold")
    tbl.add_column("Status", justify="center")

    for r in results:
        n = len(r.get("papers", []))
        err = r.get("error")
        if err:
            status = "[red]Error[/red]"
        elif n:
            status = "[green]OK[/green]"
        else:
            status = "[yellow]None found[/yellow]"
        tbl.add_row(r["topic"], str(n), status)

    console.print(tbl)


def show_paper_detail(paper: dict, index: int) -> None:
    body = (
        f"[bold cyan]Topic:[/bold cyan]   {paper.get('topic', '')}\n\n"
        f"[bold white]Title:[/bold white]   {paper.get('title', '')}\n\n"
        f"[blue]URL:[/blue]     {paper.get('url', '')}\n\n"
        f"[green]Venue:[/green]   {paper.get('venue', 'Not specified')}\n\n"
        f"[bold]Summary[/bold]\n{paper.get('summary', '')}\n\n"
        f"[bold]Relevance[/bold]\n{paper.get('relevance', '')}\n\n"
        f"[bold]Quality appraisal[/bold]\n{paper.get('quality', '')}"
    )
    console.print(
        Panel(body, title=f"Paper #{index}", border_style="cyan", expand=False)
    )


def interactive_table(all_papers: list[dict], results: list[dict]) -> None:
    if not all_papers:
        console.print("\n[yellow]No relevant papers found for any topic.[/yellow]")
        return

    console.print()
    console.rule("[bold cyan]Analysis Complete[/bold cyan]")
    console.print()
    show_topic_summary(results)
    console.print()

    current = all_papers
    title = f"All Papers — {len(all_papers)} total"

    while True:
        console.print(make_table(current, title))
        console.print(f"\n[dim]{len(current)} paper(s) displayed[/dim]\n")

        console.print(
            "[bold]Commands:[/bold]  "
            "[cyan]a[/cyan] all  "
            "[cyan]f[/cyan] filter-by-topic  "
            "[cyan]s[/cyan] search  "
            "[cyan]d[/cyan] detail  "
            "[cyan]q[/cyan] quit"
        )
        choice = Prompt.ask("→", choices=["a", "f", "s", "d", "q"], default="q")
        console.print()

        if choice == "q":
            break

        elif choice == "a":
            current = all_papers
            title = f"All Papers — {len(all_papers)} total"

        elif choice == "f":
            topics = sorted({p["topic"] for p in all_papers})
            for i, t in enumerate(topics, 1):
                console.print(f"  [cyan]{i}[/cyan]. {t}")
            raw = Prompt.ask("Topic number (0 = all)", default="0")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(topics):
                    sel = topics[idx]
                    current = [p for p in all_papers if p["topic"] == sel]
                    title = f"{sel} — {len(current)} paper(s)"
                else:
                    current = all_papers
                    title = f"All Papers — {len(all_papers)} total"
            except ValueError:
                pass

        elif choice == "s":
            kw = Prompt.ask("Keyword").strip().lower()
            if kw:
                current = [
                    p for p in all_papers
                    if any(
                        kw in (p.get(f) or "").lower()
                        for f in ("title", "summary", "topic", "venue", "relevance", "quality")
                    )
                ]
                title = f'Search "{kw}" — {len(current)} result(s)'

        elif choice == "d":
            raw = Prompt.ask("Paper number")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(current):
                    show_paper_detail(current[idx], idx + 1)
                    Prompt.ask("[dim]Press Enter to continue[/dim]")
            except (ValueError, IndexError):
                pass


# ── Output ────────────────────────────────────────────────────────────────────
def save_results(
    all_papers: list[dict], results: list[dict], digest_path: str
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — structured for programmatic analysis
    json_path = OUTPUT_DIR / f"arxiv_analysis_{ts}.json"
    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "digest_file": digest_path,
            "topics_file": str(TOPICS_FILE),
            "model": MODEL,
            "total_papers": len(all_papers),
            "topics_analysed": len(results),
        },
        "results_by_topic": results,
        "all_papers": all_papers,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown — human-readable for review
    md_path = OUTPUT_DIR / f"arxiv_analysis_{ts}.md"
    lines: list[str] = [
        "# ArXiv Digest Analysis",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Digest:** `{digest_path}`",
        f"**Model:** `{MODEL}`",
        f"**Total papers found:** {len(all_papers)}",
        f"**Topics analysed:** {len(results)}",
        "",
        "---",
        "",
    ]

    for r in results:
        topic = r.get("topic", "Unknown")
        papers = r.get("papers", [])
        err = r.get("error")
        lines += [f"## {topic}", ""]
        if err:
            lines += [f"> ⚠️ Agent error: {err}", ""]
        if not papers:
            lines += ["*No relevant papers found.*", ""]
            continue
        for i, p in enumerate(papers, 1):
            url = p.get("url", "")
            lines += [
                f"### {i}. {p.get('title', 'Untitled')}",
                "",
                f"- **URL:** [{url}]({url})",
                f"- **Venue:** {p.get('venue', 'Not specified')}",
                "",
                f"**Summary:** {p.get('summary', '')}",
                "",
                f"**Relevance to topic:** {p.get('relevance', '')}",
                "",
                f"**Quality appraisal:** {p.get('quality', '')}",
                "",
            ]
        lines += ["---", ""]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]ArXiv Digest Analyser[/bold cyan]\n"
            "Parallel LLM agents · one per topic · powered by Claude",
            border_style="cyan",
        )
    )

    digest_path = Prompt.ask("\nPath to ArXiv digest file")
    digest = read_digest(digest_path)

    topics = read_topics()
    console.print(f"\n[green]Loaded {len(topics)} topic(s) from {TOPICS_FILE}:[/green]")
    for t in topics:
        console.print(f"  [cyan]•[/cyan] {t}")

    console.print(
        f"\n[dim]Digest: {len(digest):,} characters from '{digest_path}'[/dim]"
    )
    console.print(
        f"[bold]\nSpawning {len(topics)} parallel LLM agent(s)…[/bold]"
        " [dim](digest is prompt-cached across agents)[/dim]\n"
    )

    results = asyncio.run(run_all_agents(topics, digest))

    all_papers = flatten_papers(results)
    console.print(
        f"\n[bold green]✓ Done.[/bold green] "
        f"Found [cyan bold]{len(all_papers)}[/cyan bold] relevant paper(s).\n"
    )

    json_path, md_path = save_results(all_papers, results, digest_path)
    console.print("[bold]Results saved:[/bold]")
    console.print(f"  [blue]JSON →[/blue] {json_path}")
    console.print(f"  [blue]MD   →[/blue] {md_path}")

    interactive_table(all_papers, results)
    console.print("\n[bold cyan]Bye![/bold cyan]")


if __name__ == "__main__":
    main()
