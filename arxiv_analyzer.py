#!/usr/bin/env python3
"""ArXiv Digest Analyser — Textual graphical TUI."""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button, DataTable, DirectoryTree, Footer, Header,
    Input, Label, ProgressBar, Select, Static,
)

# ── Configuration ──────────────────────────────────────────────────────────────
TOPICS_FILE = Path("topics.txt")
OUTPUT_DIR  = Path("output")
MODEL       = "claude-opus-4-6"

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert research assistant specialised in analysing ArXiv paper digests.

Rules:
- ONLY return papers that are explicitly present in the digest you are given.
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

_DIGEST_WRAPPER = """\
--- ARXIV DIGEST START ---
{digest}
--- ARXIV DIGEST END ---
"""

_TOPIC_REQUEST = """\
Identify all papers from the digest above that are relevant to this topic:

**{topic}**

Return only the JSON object.
"""

# ── Data helpers ───────────────────────────────────────────────────────────────
def get_topics() -> "list[str] | str":
    if not TOPICS_FILE.exists():
        return f"topics.txt not found"
    ts = [l.strip() for l in TOPICS_FILE.read_text("utf-8").splitlines() if l.strip()]
    return ts if ts else "topics.txt is empty"


def get_digest(path: str) -> "tuple[str | None, str | None]":
    p = Path(path)
    if not p.exists():
        return None, f"File not found: {path}"
    try:
        return p.read_text("utf-8"), None
    except Exception as exc:
        return None, str(exc)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


async def _run_agent(
    client: anthropic.AsyncAnthropic,
    topic: str,
    digest: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        try:
            text = ""
            async with client.messages.stream(
                model=MODEL,
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
                            "text": _TOPIC_REQUEST.format(topic=topic),
                        },
                    ],
                }],
            ) as stream:
                async for chunk in stream.text_stream:
                    text += chunk
            result = _extract_json(text)
            result.setdefault("papers", [])
            result["topic"] = topic
            return result
        except Exception as exc:
            return {"topic": topic, "papers": [], "error": str(exc)}


def _flatten(results: list[dict]) -> list[dict]:
    flat = []
    for r in results:
        for p in r.get("papers", []):
            flat.append({**p, "topic": r["topic"]})
    return flat


def _save(all_papers: list[dict], results: list[dict], digest_path: str):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jp = OUTPUT_DIR / f"arxiv_analysis_{ts}.json"
    jp.write_text(json.dumps({
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "digest_file": digest_path,
            "model": MODEL,
            "total_papers": len(all_papers),
        },
        "results_by_topic": results,
        "all_papers": all_papers,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    mp = OUTPUT_DIR / f"arxiv_analysis_{ts}.md"
    lines = [
        "# ArXiv Digest Analysis",
        f"\n**Date:** {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**Model:** `{MODEL}`",
        f"**Total papers:** {len(all_papers)}",
        "\n---\n",
    ]
    for r in results:
        lines.append(f"## {r['topic']}")
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


# ── CSS ────────────────────────────────────────────────────────────────────────
APP_CSS = """
/* ── Shared ── */
Screen { background: $background; }

/* ── Welcome ── */
WelcomeScreen          { align: center middle; }
#welcome-box           { width: 70; height: auto; background: $surface;
                         border: thick $primary; padding: 2 4; }
#welcome-title         { text-align: center; color: $accent; text-style: bold;
                         margin-bottom: 1; }
#welcome-sub           { text-align: center; color: $text-muted;
                         margin-bottom: 2; }
#path-row              { height: 3; margin-bottom: 1; }
#path-input            { width: 1fr; }
#btn-browse            { width: auto; margin-left: 1; }
#btn-analyse           { width: 100%; margin-top: 1; }
#welcome-err           { color: red; height: 1; }

/* ── File browser ── */
FileBrowserModal       { align: center middle; }
FileBrowserModal > Container {
    width: 72; height: 34; background: $surface;
    border: thick $primary; padding: 1 2;
}
FileBrowserModal DirectoryTree { height: 1fr; }
#fb-sel                { color: $text-muted; height: 1; margin-top: 1; }
#fb-btns               { height: 3; align: right middle; }

/* ── Processing ── */
ProcessingScreen       { align: center middle; }
#proc-box              { width: 62; height: auto; background: $surface;
                         border: thick $primary; padding: 2 4; }
#proc-title            { text-align: center; color: $accent;
                         text-style: bold; margin-bottom: 1; }
ProgressBar            { margin: 1 0; }
#proc-log              { height: 10; overflow-y: auto;
                         border: solid $primary-darken-2;
                         padding: 0 1; margin-top: 1; }

/* ── Results ── */
ResultsScreen          { layout: vertical; }
#toolbar               { height: 3; background: $boost;
                         padding: 0 1; align: left middle; }
#search                { width: 26; margin-right: 1; }
#topic-sel             { width: 36; margin-right: 1; }
#count                 { color: $text-muted; content-align: center middle;
                         width: auto; }
DataTable              { height: 1fr; }

/* ── Detail modal ── */
PaperDetailModal       { align: center middle; }
PaperDetailModal ScrollableContainer {
    width: 94; max-height: 48; background: $surface;
    border: thick $primary; padding: 1 2;
}
.dh                    { color: $accent; text-style: bold; margin-top: 1; }
.durl                  { color: $primary; }
.dvenue                { color: green; }
#d-close               { margin-top: 1; align: center middle; }
"""

# ── File Browser Modal ─────────────────────────────────────────────────────────
class FileBrowserModal(ModalScreen):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._picked: "str | None" = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("[bold]Select ArXiv Digest File[/bold]", markup=True)
            yield DirectoryTree(str(Path.cwd()))
            yield Label("No file selected", id="fb-sel")
            with Horizontal(id="fb-btns"):
                yield Button("Select", id="fb-ok", variant="primary")
                yield Button("Cancel", id="fb-cancel")

    @on(DirectoryTree.FileSelected)
    def on_file(self, e: DirectoryTree.FileSelected) -> None:
        self._picked = str(e.path)
        self.query_one("#fb-sel", Label).update(f"Selected: {Path(self._picked).name}")

    @on(Button.Pressed, "#fb-ok")
    def do_ok(self) -> None:
        self.dismiss(self._picked)

    @on(Button.Pressed, "#fb-cancel")
    def do_cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Paper Detail Modal ─────────────────────────────────────────────────────────
class PaperDetailModal(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, paper: dict) -> None:
        super().__init__()
        self._p = paper

    def compose(self) -> ComposeResult:
        p = self._p
        with ScrollableContainer():
            yield Label(p.get("topic", ""), classes="dh")
            yield Static(f"[bold]{p.get('title', '')}[/bold]", markup=True)
            yield Static(p.get("url", ""), classes="durl")
            yield Label("Venue", classes="dh")
            yield Static(p.get("venue", "Not specified"), classes="dvenue")
            yield Label("Summary", classes="dh")
            yield Static(p.get("summary", ""))
            yield Label("Relevance to Topic", classes="dh")
            yield Static(p.get("relevance", ""))
            yield Label("Quality Appraisal", classes="dh")
            yield Static(p.get("quality", ""))
            with Horizontal(id="d-close"):
                yield Button("Close  [Esc]", id="d-close-btn", variant="primary")

    @on(Button.Pressed, "#d-close-btn")
    def close(self) -> None:
        self.dismiss()


# ── Welcome Screen ─────────────────────────────────────────────────────────────
class WelcomeScreen(Screen):
    def compose(self) -> ComposeResult:
        with Container(id="welcome-box"):
            yield Label("ArXiv Digest Analyser", id="welcome-title")
            yield Label(
                "Parallel LLM agents · one per topic · powered by Claude",
                id="welcome-sub",
            )
            with Horizontal(id="path-row"):
                yield Input(placeholder="Path to digest file…", id="path-input")
                yield Button("Browse…", id="btn-browse")
            yield Button("Analyse →", id="btn-analyse", variant="primary")
            yield Label("", id="welcome-err")

    @on(Button.Pressed, "#btn-browse")
    def on_browse(self) -> None:
        self.app.push_screen(FileBrowserModal(), self._got_path)

    def _got_path(self, path: "str | None") -> None:
        if path:
            self.query_one("#path-input", Input).value = path

    @on(Button.Pressed, "#btn-analyse")
    def on_analyse(self) -> None:
        self._try_start()

    @on(Input.Submitted, "#path-input")
    def on_submit(self, _: Input.Submitted) -> None:
        self._try_start()

    def _try_start(self) -> None:
        path = self.query_one("#path-input", Input).value.strip()
        err  = self.query_one("#welcome-err", Label)
        if not path:
            err.update("Please enter a file path.")
            return
        if not Path(path).exists():
            err.update(f"File not found: {path}")
            return
        err.update("")
        self.app.begin_analysis(path)  # type: ignore[attr-defined]


# ── Processing Screen ──────────────────────────────────────────────────────────
class ProcessingScreen(Screen):
    def __init__(self, total: int) -> None:
        super().__init__()
        self._total = total
        self._lines: list[str] = []

    def compose(self) -> ComposeResult:
        with Container(id="proc-box"):
            yield Label("Analysing Digest…", id="proc-title")
            yield Label(
                f"Spawning {self._total} parallel agent(s)…", id="proc-status"
            )
            yield ProgressBar(total=self._total, show_eta=False, id="proc-bar")
            with ScrollableContainer(id="proc-log"):
                yield Static("", id="proc-text", markup=False)

    def tick(self, done: int, topic: str) -> None:
        self.query_one("#proc-bar", ProgressBar).update(progress=done)
        self.query_one("#proc-status", Label).update(
            f"Completed {done} / {self._total}…"
        )
        self._lines.append(f"✓ {topic}")
        self.query_one("#proc-text", Static).update("\n".join(self._lines))
        self.query_one("#proc-log", ScrollableContainer).scroll_end(animate=False)


# ── Results Screen ─────────────────────────────────────────────────────────────
_COL_FIELD = {
    "Topic":   "topic",
    "Title":   "title",
    "Venue":   "venue",
    "URL":     "url",
    "Summary": "summary",
}

class ResultsScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, results: list[dict], digest_path: str) -> None:
        super().__init__()
        self._results     = results
        self._digest_path = digest_path
        self._all         = _flatten(results)
        self._topics      = sorted({p["topic"] for p in self._all})
        self._shown       = list(self._all)
        self._sort_field  : "str | None" = None
        self._sort_asc    = True

    def compose(self) -> ComposeResult:
        opts = [("All topics", "ALL")] + [(t, t) for t in self._topics]
        yield Header()
        with Horizontal(id="toolbar"):
            yield Input(placeholder="🔍 Search…", id="search")
            yield Select(options=opts, value="ALL", id="topic-sel")
            yield Label("", id="count")
        yield DataTable(id="table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild()
        try:
            jp, mp = _save(self._all, self._results, self._digest_path)
            self.notify(f"Saved → {jp.name}  {mp.name}", timeout=6)
        except Exception as exc:
            self.notify(f"Save error: {exc}", severity="error")

    # ── Build / refresh table ─────────────────────────────────────────────────
    def _rebuild(self) -> None:
        dt = self.query_one("#table", DataTable)
        dt.clear(columns=True)
        dt.add_columns("#", "Topic", "Title", "Venue", "URL", "Summary")
        for i, p in enumerate(self._shown):
            s = p.get("summary", "")
            dt.add_row(
                str(i + 1),
                p.get("topic", ""),
                p.get("title", ""),
                p.get("venue", "Not specified"),
                p.get("url", ""),
                (s[:80] + "…") if len(s) > 80 else s,
                key=str(i),
            )
        self.query_one("#count", Label).update(
            f"{len(self._shown)} / {len(self._all)} papers"
        )

    def _refilter(self) -> None:
        kw    = self.query_one("#search", Input).value.strip().lower()
        sel   = self.query_one("#topic-sel", Select)
        topic = str(sel.value) if sel.value is not Select.BLANK else "ALL"

        papers = self._all
        if topic != "ALL":
            papers = [p for p in papers if p["topic"] == topic]
        if kw:
            papers = [
                p for p in papers
                if any(
                    kw in (p.get(f) or "").lower()
                    for f in ("title", "summary", "topic", "venue",
                              "relevance", "quality", "url")
                )
            ]
        if self._sort_field:
            papers = sorted(
                papers,
                key=lambda p: (p.get(self._sort_field) or "").lower(),
                reverse=not self._sort_asc,
            )
        self._shown = papers
        self._rebuild()

    # ── Events ────────────────────────────────────────────────────────────────
    @on(Input.Changed,  "#search")
    def on_search(self, _: Input.Changed) -> None:
        self._refilter()

    @on(Select.Changed, "#topic-sel")
    def on_topic(self, _: Select.Changed) -> None:
        self._refilter()

    @on(DataTable.HeaderSelected)
    def on_header(self, event: DataTable.HeaderSelected) -> None:
        field = _COL_FIELD.get(str(event.label).strip())
        if not field:
            return
        if self._sort_field == field:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_field, self._sort_asc = field, True
        self._refilter()
        arrow = "↑" if self._sort_asc else "↓"
        self.notify(f"Sorted by {event.label} {arrow}", timeout=2)

    @on(DataTable.RowSelected)
    def on_row(self, event: DataTable.RowSelected) -> None:
        try:
            paper = self._shown[int(str(event.row_key.value))]
            self.app.push_screen(PaperDetailModal(paper))
        except (ValueError, IndexError):
            pass

    def action_back(self) -> None:
        self.app.pop_screen()


# ── Application ────────────────────────────────────────────────────────────────
class ArxivAnalyzerApp(App):
    CSS      = APP_CSS
    TITLE    = "ArXiv Digest Analyser"
    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

    # called from WelcomeScreen
    def begin_analysis(self, digest_path: str) -> None:
        topics_or_err = get_topics()
        if isinstance(topics_or_err, str):
            self.notify(topics_or_err, severity="error")
            return
        topics: list[str] = topics_or_err
        self._proc = ProcessingScreen(len(topics))
        self.push_screen(self._proc)
        self._run(digest_path, topics)

    @work(exclusive=True)
    async def _run(self, digest_path: str, topics: list[str]) -> None:
        digest, err = get_digest(digest_path)
        if err:
            self.notify(err, severity="error")
            self.pop_screen()
            return

        client = anthropic.AsyncAnthropic()
        sem    = asyncio.Semaphore(8)
        done   = 0
        lock   = asyncio.Lock()

        async def one(topic: str) -> dict:
            nonlocal done
            result = await _run_agent(client, topic, digest, sem)
            async with lock:
                done += 1
                n = done
            self._proc.tick(n, topic)
            return result

        raw = await asyncio.gather(*[one(t) for t in topics],
                                   return_exceptions=True)
        results = [
            r if not isinstance(r, Exception)
            else {"topic": topics[i], "papers": [], "error": str(r)}
            for i, r in enumerate(raw)
        ]

        self.pop_screen()                                   # remove ProcessingScreen
        self.push_screen(ResultsScreen(results, digest_path))


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    ArxivAnalyzerApp().run()


if __name__ == "__main__":
    main()
