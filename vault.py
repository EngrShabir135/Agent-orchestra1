"""Obsidian-compatible memory vault.

Plain markdown files with YAML frontmatter, #tags and [[wikilinks]]. Point it at a real
Obsidian vault and both the agents and Obsidian see the same notes. No Streamlit imports,
so it can be tested and used headlessly.
"""
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

WIKI = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
TAG = re.compile(r"(?<![\w/#])#([A-Za-z][\w/-]*)")
WORD = re.compile(r"[a-z0-9_]{2,}")
STOP = set("the and for are but not you all any can had her was one our out has have this that with from they will "
           "what when who how why which their there been were your into than then them these those about would "
           "could should".split())


def tokens(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def parse(text):
    """Return (meta dict, body). Understands simple 'key: value' and 'tags: [a, b]' frontmatter."""
    meta, body = {}, text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        meta[k.strip()] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                    else:
                        meta[k.strip()] = v.strip("'\"")
            body = text[end + 4:].lstrip("\n")
    return meta, body


def build_frontmatter(meta):
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: [{', '.join(v)}]" if isinstance(v, list) else f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


class Vault:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # path -> (mtime, text)

    # ---- paths -------------------------------------------------------------
    def _p(self, rel):
        rel = str(rel).strip().replace("\\", "/").lstrip("/")
        if not rel:
            raise ValueError("empty note path")
        if not rel.lower().endswith(".md"):
            rel += ".md"
        p = (self.root / rel).resolve()
        if self.root != p and self.root not in p.parents:
            raise ValueError("path escapes the vault")
        if any(part.startswith(".") for part in p.relative_to(self.root).parts):
            raise ValueError("hidden folders (such as .obsidian) are off limits")
        return p

    def rel(self, p):
        return p.relative_to(self.root).as_posix()

    def notes(self):
        out = []
        for p in self.root.rglob("*.md"):
            parts = p.relative_to(self.root).parts
            if not any(x.startswith(".") for x in parts):
                out.append(p)
        return sorted(out)

    def _text(self, p):
        m = p.stat().st_mtime
        c = self._cache.get(p)
        if c and c[0] == m:
            return c[1]
        t = p.read_text(encoding="utf-8", errors="replace")
        self._cache[p] = (m, t)
        return t

    # ---- read / write ------------------------------------------------------
    def read(self, rel):
        p = self.resolve(rel) or self._p(rel)
        if not p.exists():
            raise FileNotFoundError(f"note not found: {rel}")
        return self._text(p)

    def write(self, rel, content, tags=None, overwrite=True):
        p = self._p(rel)
        if p.exists() and not overwrite:
            raise FileExistsError(f"note exists: {self.rel(p)}")
        p.parent.mkdir(parents=True, exist_ok=True)
        meta, body = parse(content)
        if not meta:
            meta = {"created": datetime.now().strftime("%Y-%m-%d %H:%M")}
            if tags:
                meta["tags"] = [t.lstrip("#") for t in tags]
            content = build_frontmatter(meta) + body
        p.write_text(content, encoding="utf-8")
        self._cache.pop(p, None)
        return self.rel(p)

    def append(self, rel, text):
        p = self._p(rel)
        if not p.exists():
            return self.write(rel, text)
        old = p.read_text(encoding="utf-8", errors="replace")
        p.write_text(old.rstrip("\n") + "\n\n" + text.strip() + "\n", encoding="utf-8")
        self._cache.pop(p, None)
        return self.rel(p)

    def delete(self, rel):
        p = self._p(rel)
        if p.exists():
            p.unlink()
            self._cache.pop(p, None)

    def daily(self, text=None):
        rel = f"Daily/{datetime.now():%Y-%m-%d}"
        if text is None:
            return rel
        stamp = datetime.now().strftime("%H:%M")
        return self.append(rel, f"- {stamp} {text.strip()}") if self._p(rel).exists() \
            else self.write(rel, f"# {datetime.now():%Y-%m-%d}\n\n- {stamp} {text.strip()}", tags=["daily"])

    # ---- links -------------------------------------------------------------
    def resolve(self, name):
        """Resolve a wikilink name or relative path to an existing note path."""
        name = str(name).strip().replace("\\", "/")
        try:
            p = self._p(name)
            if p.exists():
                return p
        except ValueError:
            pass
        stem = name[:-3] if name.lower().endswith(".md") else name
        stem = stem.split("/")[-1].lower()
        for p in self.notes():
            if p.stem.lower() == stem:
                return p
        return None

    def links(self, rel):
        return sorted({m.strip() for m in WIKI.findall(self.read(rel))})

    def backlinks(self, rel):
        target = self.resolve(rel)
        if not target:
            return []
        out = []
        for p in self.notes():
            if p == target:
                continue
            for name in WIKI.findall(self._text(p)):
                if self.resolve(name) == target:
                    out.append(self.rel(p))
                    break
        return out

    def graph(self):
        nodes, edges = {}, []
        for p in self.notes():
            nodes[self.rel(p)] = p.stem
            for name in set(WIKI.findall(self._text(p))):
                t = self.resolve(name)
                edges.append((self.rel(p), self.rel(t) if t else f"?{name.strip()}"))
        for _, b in edges:
            if b.startswith("?"):
                nodes.setdefault(b, b[1:] + " (missing)")
        return nodes, edges

    # ---- search (BM25 + title/tag boosts + link expansion) -----------------
    def search(self, query, k=5, expand=True):
        qt = tokens(query)
        notes = self.notes()
        if not qt or not notes:
            return []
        docs = {}
        for p in notes:
            t = self._text(p)
            meta, body = parse(t)
            tag_text = " ".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else ""
            toks = tokens(body) + tokens(p.stem) * 3 + tokens(tag_text) * 3
            docs[p] = (Counter(toks), len(toks))
        n = len(docs)
        avg = sum(d[1] for d in docs.values()) / n or 1
        df = Counter(w for c, _ in docs.values() for w in set(c))
        scored = []
        for p, (c, ln) in docs.items():
            s = 0.0
            for w in set(qt):
                if w in c:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * c[w] * 2.2 / (c[w] + 1.2 * (0.25 + 0.75 * ln / avg))
            if s > 0:
                scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        hits = [p for _, p in scored[:k]]
        if expand and hits:  # pull in the best-connected neighbour of the top hit
            for name in WIKI.findall(self._text(hits[0])):
                t = self.resolve(name)
                if t and t not in hits and len(hits) < k + 1:
                    hits.append(t)
        return [(self.rel(p), self.snippet(p, qt)) for p in hits]

    def snippet(self, p, qt, width=360):
        meta, body = parse(self._text(p))
        low = body.lower()
        pos = min([i for i in (low.find(w) for w in qt) if i >= 0] or [0])
        start = max(0, pos - 80)
        return re.sub(r"\s+", " ", body[start:start + width]).strip()

    def recent(self, days=7, limit=20):
        cut = datetime.now() - timedelta(days=days)
        rows = [(p.stat().st_mtime, p) for p in self.notes() if datetime.fromtimestamp(p.stat().st_mtime) >= cut]
        rows.sort(key=lambda x: -x[0])
        return [self.rel(p) for _, p in rows[:limit]]

    def all_tags(self):
        c = Counter()
        for p in self.notes():
            meta, body = parse(self._text(p))
            for t in (meta.get("tags") if isinstance(meta.get("tags"), list) else []):
                c[t] += 1
            for t in TAG.findall(body):
                c[t] += 1
        return c


def slug(title, limit=80):
    s = re.sub(r"[\\/:*?\"<>|#^\[\]]", " ", title)
    return re.sub(r"\s+", " ", s).strip()[:limit] or "Untitled"
