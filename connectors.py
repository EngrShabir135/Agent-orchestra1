"""Tools the agents can call, and the application connectors behind them.

No Streamlit imports. app.py fills in the CTX object once per run.

Connection types:  webhook (Slack, Discord, Mattermost, n8n, Make, Zapier), telegram, smtp (email),
github (also Gitea / Forgejo), rest (any HTTP API: Home Assistant, Nextcloud, Notion, n8n API ...),
mcp (any Model Context Protocol server), searxng (self-hosted search).
Secrets (tokens/passwords) are never stored in the database: they come from the session or from
environment variables named CONN_<NAME>_SECRET.
"""
import ipaddress
import json
import re
import smtplib
import socket
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from email.message import EmailMessage
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import requests

import vault as vaultlib

UA = {"User-Agent": "AgentOrchestra/2.0"}

CONNECTION_TYPES = {
    "webhook": dict(label="Webhook (Slack, Discord, Mattermost, n8n, Make, Zapier)", fields=["url"], secret=False),
    "telegram": dict(label="Telegram bot", fields=["chat_id"], secret=True),
    "smtp": dict(label="Email (SMTP)", fields=["host", "port", "user", "from_addr", "default_to"], secret=True),
    "github": dict(label="GitHub / Gitea / Forgejo", fields=["base", "repo"], secret=True),
    "rest": dict(label="Any REST API (Home Assistant, Nextcloud, Notion, n8n API ...)",
                 fields=["base", "auth_header", "default_headers"], secret=True),
    "mcp": dict(label="MCP server (open Model Context Protocol)", fields=["url"], secret=True),
    "searxng": dict(label="SearXNG (self-hosted web search)", fields=["base"], secret=False),
}


class Ctx:
    vault = None            # vault.Vault
    work = Path("workspace")
    secret = staticmethod(lambda name: "")        # name -> secret string
    connections = staticmethod(lambda: [])        # -> list of dict(name,type,config)
    setting = staticmethod(lambda key, default="": default)
    remember = staticmethod(lambda text: None)    # save to shared fact memory


CTX = Ctx()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def clip(s, n=6000):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def public_url_ok(url):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except socket.gaierror:
        return False
    return True


def safe_get(url, timeout=30, max_redirects=3, **kw):
    """GET that only touches public addresses, re-checking every redirect hop."""
    for _ in range(max_redirects + 1):
        if not public_url_ok(url):
            raise ValueError("blocked: only public http(s) addresses are allowed")
        r = requests.get(url, timeout=timeout, allow_redirects=False, headers=UA, **kw)
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
            url = urljoin(url, r.headers["location"])
            continue
        return r
    raise ValueError("too many redirects")


def html_to_text(html):
    html = re.sub(r"(?is)<(script|style|noscript|svg|head).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def conn_row(ctype, name=None):
    rows = [c for c in CTX.connections() if c["type"] == ctype]
    if name:
        rows = [c for c in rows if c["name"].lower() == str(name).lower()]
    if not rows:
        raise RuntimeError(f"no '{ctype}' connection{' named ' + str(name) if name else ''} is configured (Connections page)")
    return rows[0]


def secret_of(c):
    return CTX.secret(c["name"])


# ----------------------------------------------------------------------------
# workspace + vault tools
# ----------------------------------------------------------------------------
def safe_path(rel):
    p = (CTX.work / rel).resolve()
    if p != CTX.work.resolve() and CTX.work.resolve() not in p.parents:
        raise ValueError("path escapes the workspace")
    return p


def t_write_file(a):
    p = safe_path(a["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(a["content"]), encoding="utf-8")
    return f"wrote {a['path']} ({len(str(a['content']))} chars)"


def t_read_file(a):
    return clip(safe_path(a["path"]).read_text(encoding="utf-8", errors="replace"))


def t_list_files(a):
    return "\n".join(str(f.relative_to(CTX.work)) for f in sorted(CTX.work.rglob("*"))
                     if f.is_file() and not f.name.startswith(".")) or "workspace is empty"


def t_vault_search(a):
    hits = CTX.vault.search(str(a["query"]), k=int(a.get("limit", 5)))
    return "\n".join(f"[[{Path(r).stem}]] ({r}): {s}" for r, s in hits) or "no matching notes"


def t_vault_read(a):
    rel = a["note"]
    txt = clip(CTX.vault.read(rel))
    bl = CTX.vault.backlinks(rel)
    return txt + (f"\n\nBacklinks: {', '.join('[[' + Path(b).stem + ']]' for b in bl)}" if bl else "")


def t_vault_write(a):
    rel = CTX.vault.write(a["note"], str(a["content"]), tags=a.get("tags"), overwrite=bool(a.get("overwrite", True)))
    return f"wrote note {rel}"


def t_vault_append(a):
    return f"appended to {CTX.vault.append(a['note'], str(a['content']))}"


def t_vault_backlinks(a):
    return "\n".join(CTX.vault.backlinks(a["note"])) or "no backlinks"


def t_daily_note(a):
    return f"added to {CTX.vault.daily(str(a['content']))}"


def t_save_memory(a):
    CTX.remember(str(a["content"]))
    return "saved to shared fact memory"


# ----------------------------------------------------------------------------
# web tools
# ----------------------------------------------------------------------------
def t_fetch_url(a):
    r = safe_get(a["url"])
    body = r.text if "html" not in r.headers.get("content-type", "") else html_to_text(r.text)
    return f"status {r.status_code}\n{clip(body)}"


def t_web_search(a):
    query, limit = str(a["query"]), int(a.get("limit", 6))
    try:
        c = conn_row("searxng")
        r = requests.get(c["config"]["base"].rstrip("/") + "/search", params={"q": query, "format": "json"},
                         timeout=30, headers=UA)
        r.raise_for_status()
        res = r.json().get("results", [])[:limit]
        return "\n".join(f"{i + 1}. {x.get('title')}\n   {x.get('url')}\n   {x.get('content', '')[:200]}"
                         for i, x in enumerate(res)) or "no results"
    except RuntimeError:
        pass  # no searxng connection configured, fall back to DuckDuckGo
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=UA, timeout=30)
    r.raise_for_status()
    out = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=class="result__a"|$)', r.text, re.S):
        href, title, rest = m.group(1), html_to_text(m.group(2)), m.group(3)
        if "uddg=" in href:
            href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
        out.append(f"{len(out) + 1}. {title}\n   {href}\n   {html_to_text(sn.group(1))[:200] if sn else ''}")
        if len(out) >= limit:
            break
    return "\n".join(out) or "no results (search engine may have blocked the request; try fetch_url on a known page)"


def t_rss_read(a):
    r = safe_get(a["url"])
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag in ("item", "entry"):
            def f(n):
                for ch in it:
                    if ch.tag.split("}")[-1] == n:
                        return (ch.attrib.get("href") if n == "link" and ch.attrib.get("href") else ch.text) or ""
                return ""
            items.append(f"- {f('title').strip()} | {f('link').strip()} | {html_to_text(f('description') or f('summary'))[:200]}")
    return "\n".join(items[:int(a.get("limit", 10))]) or "no items found"


def t_sql_query(a):
    p = safe_path(a["db"])
    sql = str(a["sql"]).strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with|pragma\s+table_info)\b", sql):
        return "only SELECT queries are allowed"
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        cur = c.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(50)
    finally:
        c.close()
    return clip(" | ".join(cols) + "\n" + "\n".join(" | ".join(str(x) for x in r) for r in rows))


# ----------------------------------------------------------------------------
# application connectors
# ----------------------------------------------------------------------------
def t_send_webhook(a):
    c = conn_row("webhook", a.get("connection"))
    url = c["config"]["url"]
    r = requests.post(url, json={"text": a["text"], "content": a["text"]}, timeout=30)
    return f"webhook '{c['name']}' responded {r.status_code}"


def t_send_telegram(a):
    c = conn_row("telegram", a.get("connection"))
    r = requests.post(f"https://api.telegram.org/bot{secret_of(c)}/sendMessage", timeout=30,
                      json={"chat_id": c["config"]["chat_id"], "text": str(a["text"])[:4000]})
    return f"telegram responded {r.status_code}: {r.text[:200]}"


def t_send_email(a):
    c = conn_row("smtp", a.get("connection"))
    cfg = c["config"]
    msg = EmailMessage()
    msg["Subject"] = a.get("subject", "(no subject)")
    msg["From"] = cfg.get("from_addr") or cfg.get("user")
    msg["To"] = a.get("to") or cfg.get("default_to")
    if not msg["To"]:
        raise RuntimeError("no recipient")
    msg.set_content(str(a["body"]))
    port = int(cfg.get("port") or 587)
    if port == 465:
        s = smtplib.SMTP_SSL(cfg["host"], port, timeout=30)
    else:
        s = smtplib.SMTP(cfg["host"], port, timeout=30)
        s.starttls()
    try:
        if cfg.get("user"):
            s.login(cfg["user"], secret_of(c))
        s.send_message(msg)
    finally:
        s.quit()
    return f"email sent to {msg['To']}"


def _gh(c, method, path, **kw):
    base = (c["config"].get("base") or "https://api.github.com").rstrip("/")
    h = {"Accept": "application/json", "Authorization": f"token {secret_of(c)}"}
    r = requests.request(method, f"{base}/repos/{c['config']['repo']}{path}", headers=h, timeout=30, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()


def t_github_list_issues(a):
    c = conn_row("github", a.get("connection"))
    d = _gh(c, "GET", "/issues", params={"state": a.get("state", "open"), "per_page": 20})
    return "\n".join(f"#{i['number']} [{i.get('state')}] {i['title']} - {(i.get('body') or '')[:150]}"
                     for i in d if "pull_request" not in i) or "no issues"


def t_github_create_issue(a):
    c = conn_row("github", a.get("connection"))
    d = _gh(c, "POST", "/issues", json={"title": a["title"], "body": a.get("body", "")})
    return f"created issue #{d['number']}: {d.get('html_url')}"


def t_github_comment(a):
    c = conn_row("github", a.get("connection"))
    d = _gh(c, "POST", f"/issues/{int(a['number'])}/comments", json={"body": a["body"]})
    return f"commented: {d.get('html_url')}"


def t_github_read_file(a):
    c = conn_row("github", a.get("connection"))
    import base64
    d = _gh(c, "GET", f"/contents/{quote_plus(str(a['path']), safe='/')}")
    return clip(base64.b64decode(d["content"]).decode("utf-8", "replace"))


def t_rest_call(a):
    c = conn_row("rest", a.get("connection"))
    path = str(a.get("path", "/"))
    if "://" in path or path.startswith("//"):
        raise ValueError("path must be relative to the connection base URL")
    cfg = c["config"]
    headers = {}
    try:
        headers.update(json.loads(cfg.get("default_headers") or "{}"))
    except Exception:
        pass
    if secret_of(c):
        hdr = cfg.get("auth_header") or "Authorization: Bearer"
        if ":" in hdr and hdr.strip().endswith(("Bearer", "Token", "token", "Basic")):
            name, scheme = hdr.rsplit(":", 1)[0], hdr.rsplit(":", 1)[1].strip()
            headers[name.strip()] = f"{scheme} {secret_of(c)}"
        else:
            headers[hdr.strip().rstrip(":")] = secret_of(c)
    method = str(a.get("method", "GET")).upper()
    r = requests.request(method, cfg["base"].rstrip("/") + "/" + path.lstrip("/"), headers=headers, timeout=60,
                         params=a.get("query"), json=a.get("json"))
    return f"status {r.status_code}\n{clip(r.text)}"


# ---- MCP (streamable HTTP, JSON-RPC 2.0) ------------------------------------
def _mcp_parse(r):
    if "text/event-stream" in r.headers.get("content-type", ""):
        last = None
        for line in r.text.splitlines():
            if line.startswith("data:"):
                try:
                    obj = json.loads(line[5:].strip())
                except Exception:
                    continue
                if "result" in obj or "error" in obj:
                    last = obj
        return last or {}
    return r.json() if r.text.strip() else {}


def mcp_session(c):
    url = c["config"]["url"]
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if secret_of(c):
        h["Authorization"] = f"Bearer {secret_of(c)}"
    r = requests.post(url, headers=h, timeout=60, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "agent-orchestra", "version": "2.0"}}})
    r.raise_for_status()
    sid = r.headers.get("Mcp-Session-Id")
    if sid:
        h["Mcp-Session-Id"] = sid
    requests.post(url, headers=h, timeout=30, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return url, h


def mcp_rpc(c, method, params=None):
    url, h = mcp_session(c)
    r = requests.post(url, headers=h, timeout=120, json={"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}})
    r.raise_for_status()
    obj = _mcp_parse(r)
    if "error" in obj:
        raise RuntimeError(f"MCP error: {obj['error']}")
    return obj.get("result", {})


def t_mcp_list_tools(a):
    c = conn_row("mcp", a.get("connection"))
    tools = mcp_rpc(c, "tools/list").get("tools", [])
    return "\n".join(f"- {t['name']}: {t.get('description', '')[:160]} | args: "
                     f"{json.dumps(t.get('inputSchema', {}).get('properties', {}))[:300]}" for t in tools) or "no tools"


def t_mcp_call(a):
    c = conn_row("mcp", a.get("connection"))
    res = mcp_rpc(c, "tools/call", {"name": a["tool"], "arguments": a.get("arguments", {})})
    parts = [x.get("text", "") for x in res.get("content", []) if x.get("type") == "text"]
    return clip("\n".join(parts) or json.dumps(res)[:3000])


# ---- code execution ---------------------------------------------------------
def t_run_python(a):
    if CTX.setting("allow_code", "0") != "1":
        return "code execution is disabled. Enable it on the Connections page (Safety and runtime)."
    run_dir = CTX.work / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    script = run_dir / "script.py"
    if a.get("path"):
        script = safe_path(a["path"])
    else:
        script.write_text(str(a["code"]), encoding="utf-8")
    import os
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run([sys.executable, str(script)], cwd=str(CTX.work), env=env, capture_output=True,
                           text=True, timeout=int(CTX.setting("code_timeout", "30")))
    except subprocess.TimeoutExpired:
        return "timed out"
    return clip(f"exit code {p.returncode}\n--- stdout ---\n{p.stdout[-3000:]}\n--- stderr ---\n{p.stderr[-2000:]}", 5000)


# ----------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------
def _rest_risk(a):
    return "safe" if str(a.get("method", "GET")).upper() == "GET" else "risky"


# name -> (function, risk, group, usage line)
TOOLS = {
    "list_files": (t_list_files, "safe", "files", '{}'),
    "read_file": (t_read_file, "safe", "files", '{"path":"folder/name.ext"}'),
    "write_file": (t_write_file, "safe", "files", '{"path":"folder/name.ext","content":"full content"}'),
    "vault_search": (t_vault_search, "safe", "memory", '{"query":"words to look for","limit":5}'),
    "vault_read": (t_vault_read, "safe", "memory", '{"note":"Note name or path"}'),
    "vault_write": (t_vault_write, "safe", "memory", '{"note":"Folder/Title","content":"markdown with [[wikilinks]] and #tags","tags":["tag"]}'),
    "vault_append": (t_vault_append, "safe", "memory", '{"note":"Folder/Title","content":"text to add"}'),
    "vault_backlinks": (t_vault_backlinks, "safe", "memory", '{"note":"Note name"}'),
    "daily_note": (t_daily_note, "safe", "memory", '{"content":"log line for today"}'),
    "save_memory": (t_save_memory, "safe", "memory", '{"content":"one durable fact"}'),
    "web_search": (t_web_search, "safe", "web", '{"query":"search terms","limit":6}'),
    "fetch_url": (t_fetch_url, "safe", "web", '{"url":"https://example.org/page"}'),
    "rss_read": (t_rss_read, "safe", "web", '{"url":"https://example.org/feed.xml","limit":10}'),
    "sql_query": (t_sql_query, "safe", "data", '{"db":"data.db","sql":"SELECT ..."}'),
    "send_webhook": (t_send_webhook, "risky", "apps", '{"text":"message","connection":"optional name"}'),
    "send_telegram": (t_send_telegram, "risky", "apps", '{"text":"message","connection":"optional name"}'),
    "send_email": (t_send_email, "risky", "apps", '{"to":"a@b.com","subject":"...","body":"...","connection":"optional"}'),
    "github_list_issues": (t_github_list_issues, "safe", "apps", '{"state":"open","connection":"optional"}'),
    "github_read_file": (t_github_read_file, "safe", "apps", '{"path":"README.md","connection":"optional"}'),
    "github_create_issue": (t_github_create_issue, "risky", "apps", '{"title":"...","body":"..."}'),
    "github_comment": (t_github_comment, "risky", "apps", '{"number":12,"body":"..."}'),
    "rest_call": (t_rest_call, _rest_risk, "apps", '{"connection":"name","method":"GET","path":"/relative/path","query":{},"json":{}}'),
    "mcp_list_tools": (t_mcp_list_tools, "safe", "apps", '{"connection":"name"}'),
    "mcp_call": (t_mcp_call, "risky", "apps", '{"connection":"name","tool":"tool_name","arguments":{}}'),
    "run_python": (t_run_python, "risky", "code", '{"code":"print(1)"}  or  {"path":"script.py"}'),
}

RESEARCH_TOOLS = ["vault_search", "vault_read", "vault_backlinks", "web_search", "fetch_url", "rss_read",
                  "list_files", "read_file", "sql_query", "github_list_issues", "github_read_file", "mcp_list_tools"]


def risk_of(name, args):
    r = TOOLS[name][1]
    return r(args) if callable(r) else r


def tools_doc(allowed):
    lines = ["Reply with exactly ONE JSON object per turn and nothing else. Available tools:"]
    for n in allowed:
        if n in TOOLS:
            lines.append(f'{{"tool":"{n}","args":{TOOLS[n][3]}}}')
    lines.append('When the work is complete reply: {"final":"what you did and which notes/files you produced"}')
    conns = CTX.connections()
    if conns:
        lines.append("Configured connections: " + ", ".join(f"{c['name']} ({c['type']})" for c in conns))
    return "\n".join(lines)


def call_tool(name, args):
    """Run a tool directly (no approval logic). Errors are returned as text so the agent can adapt."""
    if name not in TOOLS:
        return f"unknown tool: {name}. Use one of: {', '.join(TOOLS)}"
    if not isinstance(args, dict):
        return "tool error: args must be a JSON object"
    try:
        return str(TOOLS[name][0](args))
    except KeyError as e:
        return f"tool error: missing argument {e}. Usage: {TOOLS[name][3]}"
    except Exception as e:
        return f"tool error: {type(e).__name__}: {e}"


def test_connection(c):
    t, cfg = c["type"], c["config"]
    try:
        if t == "webhook":
            return True, "URL saved. Use 'Send test message' to verify delivery." if cfg.get("url", "").startswith("http") else "URL must start with http"
        if t == "telegram":
            r = requests.get(f"https://api.telegram.org/bot{secret_of(c)}/getMe", timeout=20)
            return r.ok, r.text[:200]
        if t == "github":
            d = _gh(c, "GET", "")
            return True, f"repo {d.get('full_name')} reachable"
        if t == "rest":
            r = requests.get(cfg["base"], timeout=20)
            return r.status_code < 500, f"status {r.status_code}"
        if t == "mcp":
            tools = mcp_rpc(c, "tools/list").get("tools", [])
            return True, f"{len(tools)} tools: " + ", ".join(x["name"] for x in tools[:8])
        if t == "searxng":
            r = requests.get(cfg["base"].rstrip("/") + "/search", params={"q": "test", "format": "json"}, timeout=20)
            return r.ok, f"status {r.status_code}"
        if t == "smtp":
            port = int(cfg.get("port") or 587)
            s = smtplib.SMTP_SSL(cfg["host"], port, timeout=20) if port == 465 else smtplib.SMTP(cfg["host"], port, timeout=20)
            if port != 465:
                s.starttls()
            if cfg.get("user"):
                s.login(cfg["user"], secret_of(c))
            s.quit()
            return True, "login ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return False, "unknown type"
