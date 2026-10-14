"""AI Agent Orchestra - multi-provider agent platform built with Streamlit.

CEO agent -> 4 specialist agents -> Executor agent (acts through real tools and connections).
Works with local Ollama out of the box. Obsidian-compatible memory vault (vault.py), application
connectors + MCP + approval queue (connectors.py), playbooks, wallet budgets, app building.
"""
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import connectors as cx
import vault as vaultlib

BASE_DIR = Path(__file__).parent
DATA = BASE_DIR / "data"
WORK = DATA / "workspace"
WORK.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "orchestra.db"

# ----------------------------------------------------------------------------
# Provider registry. kind: anthropic | gemini | openai (OpenAI-compatible) | yandex
# Prices are USD per 1M tokens and are editable in the app: confirm them on each
# provider's pricing page.
# ----------------------------------------------------------------------------
PROVIDERS = {
    "Anthropic Claude": dict(kind="anthropic", origin="USA", env="ANTHROPIC_API_KEY",
                             base="https://api.anthropic.com/v1", model="claude-sonnet-5-5", pin=3.0, pout=15.0),
    "Google Gemini": dict(kind="gemini", origin="USA", env="GEMINI_API_KEY",
                          base="https://generativelanguage.googleapis.com/v1beta", model="gemini-2.5-flash", pin=0.30, pout=2.50),
    "OpenAI": dict(kind="openai", origin="USA", env="OPENAI_API_KEY",
                   base="https://api.openai.com/v1", model="gpt-4o-mini", pin=0.15, pout=0.60),
    "Groq (open models)": dict(kind="openai", origin="USA", env="GROQ_API_KEY",
                               base="https://api.groq.com/openai/v1", model="llama-3.3-70b-versatile", pin=0.59, pout=0.79),
    "OpenRouter": dict(kind="openai", origin="Aggregator", env="OPENROUTER_API_KEY",
                       base="https://openrouter.ai/api/v1", model="meta-llama/llama-3.3-70b-instruct", pin=0.10, pout=0.30),
    "Mistral AI": dict(kind="openai", origin="France", env="MISTRAL_API_KEY",
                       base="https://api.mistral.ai/v1", model="mistral-large-latest", pin=2.0, pout=6.0),
    "DeepSeek": dict(kind="openai", origin="China", env="DEEPSEEK_API_KEY",
                     base="https://api.deepseek.com/v1", model="deepseek-chat", pin=0.28, pout=0.42),
    "Alibaba Qwen": dict(kind="openai", origin="China", env="DASHSCOPE_API_KEY",
                         base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", model="qwen-plus", pin=0.40, pout=1.20),
    "Moonshot Kimi": dict(kind="openai", origin="China", env="MOONSHOT_API_KEY",
                          base="https://api.moonshot.ai/v1", model="moonshot-v1-8k", pin=0.20, pout=2.0),
    "Zhipu GLM": dict(kind="openai", origin="China", env="ZHIPU_API_KEY",
                      base="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash", pin=0.0, pout=0.0),
    "YandexGPT": dict(kind="yandex", origin="Russia", env="YANDEX_API_KEY",
                      base="https://llm.api.cloud.yandex.net/v1", model="gpt://YOUR_FOLDER_ID/yandexgpt/latest", pin=0.0, pout=0.0),
    "Ollama (local open source)": dict(kind="ollama", origin="Open source / local", env="OLLAMA_LOCAL_KEY", nokey=True,
                                      base=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), model="llama3.2", pin=0.0, pout=0.0),
    "Ollama Cloud (online API)": dict(kind="ollama", origin="Open source / local", env="OLLAMA_API_KEY",
                                      base="https://ollama.com", model="gpt-oss:120b", pin=0.0, pout=0.0),
}

DEFAULT_AGENTS = [
    ("CEO", "ceo", "Main orchestrator. Sets targets, plans work, assigns it to specialists, reviews results and drives growth.",
     "You are the CEO agent of an AI organisation. You think in goals, measurable targets and growth. You plan clearly, delegate precisely and judge results honestly."),
    ("Researcher", "specialist", "Finds, verifies and structures information.",
     "You are the Researcher agent. Produce accurate, structured findings. State clearly what you are certain of and what needs verification."),
    ("Developer", "specialist", "Designs and writes working software.",
     "You are the Developer agent. Write complete, working, well-structured code and explain how to run it."),
    ("Analyst", "specialist", "Analyses data, risks, costs and priorities.",
     "You are the Analyst agent. Evaluate options with numbers, trade-offs and clear recommendations."),
    ("Writer", "specialist", "Produces clear documents, copy and communications.",
     "You are the Writer agent. Produce polished, clear, ready-to-use text in the requested format."),
    ("Executor", "executor", "Performs actions: writes files, fetches web pages, saves memory, sends webhooks.",
     "You are the Executor agent. You carry out the plan by calling tools, one action at a time, and report exactly what was done."),
]

# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
SCHEMA = """
create table if not exists agents(id integer primary key autoincrement, name text unique, kind text, role text,
  system_prompt text, provider text, model text, budget_usd real default 5, spent_usd real default 0);
create table if not exists memory(id integer primary key autoincrement, agent_id integer default 0, content text, created text default current_timestamp);
create table if not exists tasks(id integer primary key autoincrement, title text, description text, agent_id integer,
  status text default 'todo', result text, created text default current_timestamp, finished text);
create table if not exists goals(id integer primary key autoincrement, title text, metric text, progress integer default 0, note text, created text default current_timestamp);
create table if not exists usage(id integer primary key autoincrement, agent_id integer, provider text, model text, tokens_in integer, tokens_out integer, cost real, ts text default current_timestamp);
create table if not exists runs(id integer primary key autoincrement, goal text, plan text, summary text, status text, created text default current_timestamp);
create table if not exists providers(name text primary key, base text, model text, pin real, pout real);
create table if not exists settings(key text primary key, value text);
create table if not exists custom_providers(name text primary key, origin text, nokey integer default 0);
create table if not exists connections(name text primary key, type text, config text);
create table if not exists actions(id integer primary key autoincrement, run_id integer, agent text, tool text, args text, status text, result text, created text default current_timestamp);
create table if not exists playbooks(name text primary key, description text, template text);
"""


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def q(sql, args=()):
    c = conn()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return cur.fetchall()
    finally:
        c.close()


def ex(sql, args=()):
    c = conn()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return cur.lastrowid
    finally:
        c.close()


def init_db():
    c = conn()
    c.executescript(SCHEMA)
    for n, p in PROVIDERS.items():
        c.execute("insert or ignore into providers values(?,?,?,?,?)", (n, p["base"], p["model"], p["pin"], p["pout"]))
    if c.execute("select count(*) from agents").fetchone()[0] == 0:
        dp = OLLAMA
        for name, kind, role, sp in DEFAULT_AGENTS:
            c.execute("insert into agents(name,kind,role,system_prompt,provider,model) values(?,?,?,?,?,?)",
                      (name, kind, role, sp, dp, PROVIDERS[dp]["model"]))
    c.execute("update providers set base=? where name=? and base like '%/v1'", (PROVIDERS[OLLAMA]["base"], OLLAMA))
    for n, d, t in PLAYBOOKS:
        c.execute("insert or ignore into playbooks values(?,?,?)", (n, d, t))
    row = c.execute("select value from settings where key='webhook_url'").fetchone()
    if row and row[0] and not c.execute("select 1 from connections where type='webhook'").fetchone():
        c.execute("insert into connections values('default-webhook','webhook',?)", (json.dumps({"url": row[0]}),))
    if not c.execute("select 1 from settings where key='ollama_default_applied'").fetchone():
        if c.execute("select count(*) from usage").fetchone()[0] == 0:  # nothing spent yet: move default agents to local Ollama
            c.execute("update agents set provider=?, model=? where provider='Anthropic Claude'", (OLLAMA, PROVIDERS[OLLAMA]["model"]))
        c.execute("insert into settings values('ollama_default_applied','1')")
    c.commit()
    c.close()


def setting(key, default=""):
    r = q("select value from settings where key=?", (key,))
    return r[0]["value"] if r else default


def set_setting(key, value):
    ex("insert into settings(key,value) values(?,?) on conflict(key) do update set value=excluded.value", (key, value))


# ----------------------------------------------------------------------------
# LLM layer
# ----------------------------------------------------------------------------
ORIGIN_ORDER = ["USA", "France", "China", "Russia", "Open source / local", "Aggregator"]


def registry():
    """Built-in platforms plus any platform the user added (OpenAI-compatible)."""
    reg = dict(PROVIDERS)
    for r in q("select c.name, c.origin, c.nokey, p.base, p.model, p.pin, p.pout from custom_providers c join providers p on p.name=c.name"):
        env = "CUSTOM_" + re.sub(r"[^A-Z0-9]", "_", r["name"].upper()) + "_API_KEY"
        reg[r["name"]] = dict(kind="openai", origin=r["origin"] or "Custom", env=env, nokey=bool(r["nokey"]),
                              base=r["base"], model=r["model"], pin=r["pin"], pout=r["pout"])
    return reg


def origins_in(reg):
    found = {p["origin"] for p in reg.values()}
    return [o for o in ORIGIN_ORDER if o in found] + sorted(found - set(ORIGIN_ORDER))


def prov(name):
    row = dict(q("select * from providers where name=?", (name,))[0])
    row.update({k: v for k, v in registry()[name].items() if k in ("kind", "origin", "env", "nokey")})
    return row


def get_key(name):
    return st.session_state.setdefault("keys", {}).get(name) or os.environ.get(registry()[name]["env"], "")


def _check(r, name):
    if r.status_code >= 400:
        raise RuntimeError(f"{name} returned {r.status_code}: {r.text[:400]}")


OLLAMA = "Ollama (local open source)"


def ollama_host(base):
    b = (base or "").strip().rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    if b and not b.startswith("http"):
        b = "http://" + b
    return b


@st.cache_data(ttl=10, show_spinner=False)
def _ollama_tags(host, key):
    r = requests.get(host + "/api/tags", timeout=4, headers={"Authorization": f"Bearer {key}"} if key else {})
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def ollama_models(name=OLLAMA):
    """(installed model names, error text). Never raises."""
    try:
        return _ollama_tags(ollama_host(prov(name)["base"]), get_key(name)), ""
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def llm_raw(name, model, system, prompt, max_tokens=3000, json_mode=False):
    p = prov(name)
    key = get_key(name)
    if not key and not registry()[name].get("nokey"):
        raise RuntimeError(f"No API key set for {name}. Add it on the Providers page or set {p['env']}.")
    base = p["base"].rstrip("/")
    kind = p["kind"]
    if kind == "ollama":
        host = ollama_host(base)
        payload = {"model": model, "stream": False, "keep_alive": "15m",
                   "options": {"num_predict": max_tokens, "num_ctx": int(setting("ollama_ctx", "8192")),
                               "temperature": float(setting("ollama_temp", "0.3"))},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(host + "/api/chat", json=payload, timeout=int(setting("ollama_timeout", "900")),
                              headers={"Authorization": f"Bearer {key}"} if key else {})
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Ollama at {host}. Start it with 'ollama serve' (or open the Ollama app), "
                               "or fix the address on the Platform settings page.")
        if r.status_code in (401, 403):
            raise RuntimeError(f"Ollama refused the request ({r.status_code}). " + (
                "Add your Ollama API key on the API keys page (create one at https://ollama.com/settings/keys)." if "ollama.com" in host
                else "Online ':cloud' models need you to sign in first: run 'ollama signin' in a terminal."))
        if r.status_code == 404:
            fix = f"ollama signin   then   ollama pull {model}" if is_cloud_tag(model) else f"ollama pull {model}"
            raise RuntimeError(f"Ollama does not have the model '{model}'. Install it with: {fix}")
        _check(r, name)
        d = r.json()
        return d.get("message", {}).get("content", ""), d.get("prompt_eval_count", 0), d.get("eval_count", 0)
    if kind == "anthropic":
        r = requests.post(base + "/messages", timeout=180,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": model, "max_tokens": max_tokens, "system": system,
                                "messages": [{"role": "user", "content": prompt}]})
        _check(r, name)
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)
    if kind == "gemini":
        r = requests.post(f"{base}/models/{model}:generateContent", timeout=180,
                          headers={"x-goog-api-key": key, "content-type": "application/json"},
                          json={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                "generationConfig": {"maxOutputTokens": max_tokens}})
        _check(r, name)
        d = r.json()
        parts = d["candidates"][0].get("content", {}).get("parts", [])
        u = d.get("usageMetadata", {})
        return "".join(x.get("text", "") for x in parts), u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    headers = {"content-type": "application/json"}
    if key:
        headers["Authorization"] = f"Api-Key {key}" if kind == "yandex" else f"Bearer {key}"
    if "openrouter.ai" in base:
        headers["HTTP-Referer"] = "http://localhost:8501"
        headers["X-Title"] = "AI Agent Orchestra"
    r = requests.post(base + "/chat/completions", timeout=180, headers=headers,
                      json={"model": model, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    _check(r, name)
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def recall(agent_id, query="", limit=10):
    """Short facts + the vault notes most relevant to the current query (kept small for local models)."""
    parts = []
    rows = q("select content from memory where agent_id in (0,?) order by id desc limit ?", (agent_id, limit))
    if rows:
        parts.append("Facts:\n" + "\n".join(f"- {r['content'][:300]}" for r in reversed(rows)))
    if query.strip():
        try:
            hits = vault().search(query[:600], k=4)
        except Exception:
            hits = []
        if hits:
            parts.append("Relevant notes from the knowledge vault:\n" +
                         "\n".join(f"[[{Path(r).stem}]]: {s[:280]}" for r, s in hits))
    return "\n\n".join(parts)


def call_llm(agent, prompt, system=None, max_tokens=3000, json_mode=False, query=None):
    agent = dict(q("select * from agents where id=?", (agent["id"],))[0])  # fresh budget numbers
    p = prov(agent["provider"])
    free = not p["pin"] and not p["pout"]
    if not free and agent["spent_usd"] >= agent["budget_usd"]:
        raise RuntimeError(f"Wallet for agent {agent['name']} is exhausted (${agent['spent_usd']:.4f} of ${agent['budget_usd']:.2f}). Allocate more on the Wallet page.")
    cap = to_int(setting("token_cap_daily", "0"))
    if cap > 0 and tokens_today() >= cap:
        raise RuntimeError(f"Daily token limit reached ({cap:,} tokens). Raise it on the Tokens page (Limits tab) to continue.")
    sys_prompt = system if system is not None else agent["system_prompt"]
    mem = recall(agent["id"], query if query is not None else prompt[:500])
    if mem:
        sys_prompt += "\n\nPersistent memory you can rely on:\n" + mem
    text, tin, tout = llm_raw(agent["provider"], agent["model"], sys_prompt, prompt, max_tokens, json_mode)
    text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()  # reasoning models (deepseek-r1, qwen3 ...)
    cost = tin / 1e6 * p["pin"] + tout / 1e6 * p["pout"]
    ex("insert into usage(agent_id,provider,model,tokens_in,tokens_out,cost) values(?,?,?,?,?,?)",
       (agent["id"], agent["provider"], agent["model"], tin, tout, cost))
    ex("update agents set spent_usd=spent_usd+? where id=?", (cost, agent["id"]))
    return text


def parse_json(text, want="object"):
    """First valid JSON object (or array) in the text; tolerates code fences and chatter around it."""
    t = re.sub(r"```(?:json)?", "", text).strip()
    opener = "{" if want == "object" else "["
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch == opener:
            try:
                obj, _ = dec.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found")


def llm_json(agent, prompt, want="object", max_tokens=2000, retries=2):
    """Ask for JSON, and re-ask with a correction when a (small, local) model gets the format wrong."""
    last = ""
    for i in range(retries + 1):
        p = prompt if i == 0 else prompt + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON, no commentary."
        last = call_llm(agent, p, max_tokens=max_tokens, json_mode=(want == "object"))
        try:
            return parse_json(last, want)
        except Exception:
            continue
    raise ValueError("the model did not return valid JSON: " + last[:200])


def to_int(v, default=0):
    try:
        return int(float(str(v).strip().rstrip("%")))
    except Exception:
        return default


def agents_by(kind=None):
    if kind:
        return [dict(r) for r in q("select * from agents where kind=? order by id", (kind,))]
    return [dict(r) for r in q("select * from agents order by id")]


def agent_by_name(name):
    r = q("select * from agents where lower(name)=lower(?)", (name,))
    return dict(r[0]) if r else None


# ----------------------------------------------------------------------------
# Vault, connections and tools (the logic lives in vault.py and connectors.py)
# ----------------------------------------------------------------------------
def vault():
    path = setting("vault_path") or str(DATA / "vault")
    cache = st.session_state.setdefault("_vault", {})
    if cache.get("path") != path:
        try:
            obj = vaultlib.Vault(path)
        except Exception:
            path, obj = str(DATA / "vault"), vaultlib.Vault(DATA / "vault")
        cache["path"], cache["obj"] = path, obj
    return cache["obj"]


def conn_secret(name):
    env = "CONN_" + re.sub(r"[^A-Z0-9]", "_", name.upper()) + "_SECRET"
    return st.session_state.setdefault("conn_secrets", {}).get(name) or os.environ.get(env, "")


def sync_ctx():
    c = cx.CTX
    c.vault, c.work = vault(), WORK
    c.secret = conn_secret
    c.connections = lambda: [dict(name=r["name"], type=r["type"], config=json.loads(r["config"] or "{}"))
                             for r in q("select * from connections order by name")]
    c.setting = setting
    c.remember = lambda t: ex("insert into memory(agent_id,content) values(0,?)", (t,))


def safe_path(rel):
    sync_ctx()
    return cx.safe_path(rel)


def execute_tool(name, args, run_id=0, who=""):
    """Run a tool for an agent. Risky tools wait for human approval unless approval mode is 'auto'."""
    sync_ctx()
    if not isinstance(args, dict):
        args = {}
    if name not in cx.TOOLS:
        return cx.call_tool(name, args)
    if cx.risk_of(name, args) == "risky" and setting("approval_mode", "ask") == "ask":
        aid = ex("insert into actions(run_id,agent,tool,args,status) values(?,?,?,?,'pending')",
                 (run_id, who, name, json.dumps(args, default=str)))
        return (f"QUEUED for human approval as action #{aid}. It has NOT happened yet. "
                "Carry on with other work and say in your final report that it awaits approval.")
    res = cx.call_tool(name, args)
    if cx.TOOLS[name][2] in ("apps", "code") or name in ("vault_write", "write_file"):
        ex("insert into actions(run_id,agent,tool,args,status,result) values(?,?,?,?,?,?)",
           (run_id, who, name, json.dumps(args, default=str)[:3000], "error" if res.startswith("tool error") else "done", res[:3000]))
    return res


def approve_action(aid):
    sync_ctx()
    a = q("select * from actions where id=?", (aid,))[0]
    res = cx.call_tool(a["tool"], json.loads(a["args"] or "{}"))
    ex("update actions set status=?, result=? where id=?", ("error" if res.startswith("tool error") else "done", res[:3000], aid))
    return res


def specialist_tools(agent):
    """Which tools a specialist may use while working. Executor gets everything; writers just write."""
    n = agent["name"].lower()
    if n == "researcher":
        return cx.RESEARCH_TOOLS
    if n == "developer":
        return cx.RESEARCH_TOOLS + ["write_file", "run_python"]
    if n == "analyst":
        return ["vault_search", "vault_read", "sql_query", "read_file", "list_files", "fetch_url"]
    return []


def agent_loop(agent, goal, material, allowed, log, run_id=0, max_steps=None):
    """Tool loop: the agent answers with one JSON action per turn until it returns {"final": ...}."""
    sync_ctx()
    max_steps = max_steps or int(setting("max_steps", "10"))
    doc = cx.tools_doc(allowed)
    hist_limit = int(setting("history_chars", "6000"))
    transcript, seen = [], {}
    for step in range(1, max_steps + 1):
        hist = "\n".join(transcript)[-hist_limit:]
        prompt = (f"Goal:\n{goal}\n\nMaterial from other agents:\n{material[:6000] or 'none'}\n\n{doc}\n\n"
                  f"Actions so far:\n{hist or 'none'}\n\nYour next JSON object:")
        out = call_llm(agent, prompt, max_tokens=3500, json_mode=True, query=goal)
        try:
            obj = parse_json(out)
        except Exception:
            transcript.append(f"[step {step}] your reply was not valid JSON. Reply with exactly one JSON object.")
            continue
        if "final" in obj:
            return str(obj["final"])
        name = obj.get("tool") or obj.get("name") or obj.get("action")
        args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
        if name not in allowed:
            res = f"tool '{name}' is not available to you. Available: {', '.join(allowed)}"
        else:
            key = name + json.dumps(args, sort_keys=True, default=str)
            seen[key] = seen.get(key, 0) + 1
            res = ("You already made this exact call. Use a different step or reply with the final answer."
                   if seen[key] > 2 else execute_tool(name, args, run_id, agent["name"]))
        log(f"{agent['name']} step {step}: {name} -> {str(res)[:160]}")
        transcript.append(f"[step {step}] {json.dumps({'tool': name, 'args': args}, default=str)[:500]}\nresult: {str(res)[:1800]}")
    return "Stopped at the step limit.\n" + "\n".join(transcript)[-1500:]


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
PLAYBOOKS = [
    ("Research brief to Obsidian", "Research a topic on the web and file a linked note in the knowledge vault.",
     "Research this topic: {topic}. Researcher: search the web and read at least 3 sources with fetch_url, then report findings "
     "with source URLs. Writer: turn the findings into a clear brief. Executor: first vault_search for related notes, then save the "
     "brief as vault note 'Research/{topic}' with tags and [[wikilinks]] to related existing notes, and add a line to today's daily note."),
    ("Daily briefing", "Read RSS feeds and recent vault activity, write today's briefing and send it out.",
     "Create today's briefing. Feeds: {feed_urls}. Researcher: read each feed with rss_read and pick the 5 most important items. "
     "Analyst: check vault_search for related earlier notes and open goals. Writer: write a concise briefing. Executor: save it as "
     "vault note 'Briefings/Briefing' and send it with send_telegram or send_webhook if a connection exists."),
    ("GitHub issue triage", "Rank open issues in the connected repository and queue a recommendation comment.",
     "Triage the open issues of the connected GitHub/Gitea repository. Researcher: list them with github_list_issues. Analyst: rank by "
     "impact and effort. Writer: write a short triage report. Executor: save it as vault note 'Triage/Issue triage' and queue a "
     "github_comment on the top issue with the recommended next step."),
    ("Build and test a tool", "Developer writes a Python tool with tests, runs them and documents the result.",
     "Build this tool in Python: {spec}. Developer: write the code and a test file in the workspace with write_file, run the tests "
     "with run_python and fix failures. Executor: write a README in the workspace and a vault note 'Projects/{name}' that summarises "
     "what was built and links to the files."),
    ("Website change monitor", "Fetch pages, compare with the last stored note and alert when something changed.",
     "Monitor these pages: {urls}. Researcher: fetch each page with fetch_url and summarise its key content. Analyst: vault_search for "
     "the previous snapshot note and list what changed. Executor: save the new snapshot as vault note 'Monitor/Snapshot' and send an "
     "alert through a webhook or Telegram connection only if something changed."),
    ("Weekly review", "Review the last week of vault notes, update goals and plan next actions.",
     "Run the weekly review. Researcher: use vault_search and vault_read on recent notes (runs, daily notes, goals). Analyst: list "
     "what was achieved, what is stuck and the three highest-value next actions. Writer: write the review. Executor: save it as vault "
     "note 'Reviews/Weekly review' linking to the notes it used."),
    ("Email draft and send", "Draft an email and queue it for your approval.",
     "Write and send an email to {to} about: {topic}. Writer: draft a clear, polite email. Executor: send it with send_email "
     "(it will wait for human approval) and log it in today's daily note."),
]


def to_text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)


def agent_note(vlt, name, role, run_note):
    rel = f"Agents/{vaultlib.slug(name)}"
    if vlt.resolve(rel) is None:
        vlt.write(rel, f"# {name}\n\n{role}\n\n## Runs\n", tags=["agent"])
    vlt.append(rel, f"- [[{run_note}]]")


def run_orchestra(goal, log):
    sync_ctx()
    vlt = vault()
    ceo = agents_by("ceo")[0]
    specialists = agents_by("specialist")[:4]
    executor = agents_by("executor")[0]
    names = ", ".join(f"{a['name']} ({a['role']})" for a in specialists)
    log("CEO is planning...")
    try:
        plan = llm_json(ceo, (
            f"Business goal:\n{goal}\n\nAvailable specialist agents: {names}\n\n"
            'Return only JSON: {"target":{"title":"measurable target","metric":"how success is measured"},'
            '"tasks":[{"agent":"exact specialist name","task":"specific instructions"}]}. '
            "Use between 1 and 6 tasks."), "object", 2000)
    except Exception as e:
        log(f"CEO plan was not valid JSON ({e}); using a single-task plan instead.")
        plan = {}
    tasks = [t for t in (plan.get("tasks") if isinstance(plan.get("tasks"), list) else []) if isinstance(t, dict)][:6]
    if not tasks:
        tasks = [{"agent": specialists[0]["name"], "task": goal}]
    plan["tasks"] = tasks
    target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
    goal_title = str(target.get("title") or goal[:120])
    goal_id = ex("insert into goals(title,metric,progress) values(?,?,0)", (goal_title, str(target.get("metric", ""))))
    run_id = ex("insert into runs(goal,plan,status) values(?,?,'running')", (goal, json.dumps(plan, indent=2, default=str)))
    outputs, used = [], []
    for t in tasks:
        ag = agent_by_name(str(t.get("agent", ""))) or specialists[0]
        task_text = to_text(t.get("task", ""))
        log(f"{ag['name']} working: {task_text[:120]}")
        context = "\n\n".join(outputs)[-8000:]
        allowed = specialist_tools(ag)
        if allowed:
            res = agent_loop(ag, f"Overall goal: {goal}\n\nYour task: {task_text}\n\nWhen done, reply with a final answer that "
                                 "contains your complete findings or work product.", context, allowed, log, run_id)
        else:
            res = call_llm(ag, f"Overall goal: {goal}\n\nYour task: {task_text}\n\nWork already done by other agents:\n{context or 'none'}",
                           query=goal)
        outputs.append(f"## {ag['name']}: {task_text}\n{res}")
        used.append(ag["name"])
        ex("insert into tasks(title,description,agent_id,status,result,finished) values(?,?,?,'done',?,current_timestamp)",
           (task_text[:100], task_text, ag["id"], res))
    material = "\n\n".join(outputs)
    log("Executor is carrying out the plan...")
    exec_summary = agent_loop(executor, goal, material, list(cx.TOOLS), log, run_id, max_steps=int(setting("max_steps", "10")) + 4)
    log("CEO is reviewing results...")
    try:
        review = llm_json(ceo, (
            f"Goal: {goal}\nTarget: {json.dumps(target, default=str)}\n\nSpecialist work:\n{material[:7000]}\n\nExecutor report:\n{exec_summary[:3000]}\n\n"
            'Return only JSON: {"progress":0-100,"summary":"honest review","growth_actions":["next actions that grow the result"],'
            '"facts":["up to 3 durable facts worth remembering for future runs"]}'), "object", 1500)
    except Exception:
        review = {"progress": 0, "summary": "The CEO review could not be parsed.", "growth_actions": [], "facts": []}
    progress = min(max(to_int(review.get("progress")), 0), 100)
    review["progress"] = progress
    actions = [to_text(a) for a in review.get("growth_actions", [])] if isinstance(review.get("growth_actions"), list) else []
    facts = [to_text(f) for f in review.get("facts", [])][:3] if isinstance(review.get("facts"), list) else []
    # ---- write the run into the Obsidian-style vault: run note <-> goal note <-> agent notes
    stamp = datetime.now().strftime("%Y-%m-%d %H%M")
    note = f"Runs/{stamp} {vaultlib.slug(goal, 50)}"
    goal_note = f"Goals/{vaultlib.slug(goal_title)}"
    body = [f"# {goal_title}", "", f"**Goal:** {goal}", f"**Metric:** {target.get('metric', '')}", f"**Progress:** {progress}%",
            f"**Goal note:** [[{goal_note}]]", "", "## Agents"] + [f"- [[Agents/{vaultlib.slug(n)}]]" for n in dict.fromkeys(used + [executor['name']])]
    body += ["", "## Work"] + [o[:3500] for o in outputs]
    body += ["", "## Executor report", exec_summary[:3000], "", "## CEO review", to_text(review.get("summary", "")), "", "## Next growth actions"]
    body += [f"- {a}" for a in actions] or ["- none"]
    try:
        vlt.write(note, "\n".join(body), tags=["run"])
        if vlt.resolve(goal_note) is None:
            vlt.write(goal_note, f"# {goal_title}\n\n{target.get('metric', '')}\n\n## History\n", tags=["goal"])
        vlt.append(goal_note, f"- {stamp}: {progress}% - [[{note}]]")
        for n in dict.fromkeys(used + [executor["name"]]):
            ag = agent_by_name(n)
            agent_note(vlt, n, ag["role"] if ag else "", note)
        if facts:
            vlt.append("Memory/Facts", "\n".join(f"- {f} (from [[{note}]])" for f in facts))
    except Exception as e:
        log(f"Could not write the vault note: {e}")
    for f in facts:
        ex("insert into memory(agent_id,content) values(0,?)", (f[:400],))
    ex("update goals set progress=?, note=? where id=?", (progress, json.dumps(actions), goal_id))
    ex("insert into memory(agent_id,content) values(?,?)", (ceo["id"], f"Run on '{goal[:80]}': {to_text(review.get('summary', ''))[:400]}"))
    review["note"] = note + ".md"
    ex("update runs set summary=?, status='done' where id=?", (json.dumps(review, indent=2, default=str), run_id))
    return plan, outputs, exec_summary, review


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_dashboard():
    st.title("AI Agent Orchestra")
    reg = registry()
    if not any(get_key(n) or reg[n].get("nokey") for n in reg):
        st.warning("No API key is set yet. Open the API keys page and add the key for at least one platform.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", q("select count(*) c from agents")[0]["c"])
    c2.metric("Tasks done", q("select count(*) c from tasks where status='done'")[0]["c"])
    c3.metric("Memory (facts / notes)", f"{q('select count(*) c from memory')[0]['c']} / {len(vault().notes())}")
    c4.metric("Total spend (USD)", f"{q('select coalesce(sum(cost),0) c from usage')[0]['c']:.4f}")
    pend = q("select count(*) c from actions where status='pending'")[0]["c"]
    if pend:
        st.warning(f"{pend} agent action(s) are waiting for your approval on the Actions page.")
    ollama_panel()
    st.subheader("Goals and growth")
    goals = q("select * from goals order by id desc limit 10")
    if not goals:
        st.info("No goals yet. Give the CEO a goal on the Orchestra page.")
    for g in goals:
        st.write(f"**{g['title']}**  \n{g['metric'] or ''}")
        st.progress(min(max(int(g["progress"]), 0), 100) / 100, text=f"{g['progress']}% achieved")
        if g["note"]:
            try:
                for a in json.loads(g["note"]):
                    st.caption(f"Next growth action: {a}")
            except Exception:
                pass
    st.subheader("Recent runs")
    runs = q("select id, goal, status, created from runs order by id desc limit 8")
    if runs:
        show_df(pd.DataFrame([dict(r) for r in runs]))


def key_status(reg, n):
    if reg[n].get("nokey"):
        return "Not required"
    k = get_key(n)
    return f"Set (ends {k[-4:]})" if k else "Missing"


def page_keys():
    st.title("API keys")
    st.caption("Enter the key for each AI platform you want to use. Keys stay in this browser session only and are never written to disk. "
               "You can also set them as environment variables (the variable name is shown next to each platform).")
    online_quick_setup()
    reg = registry()
    keys = st.session_state.setdefault("keys", {})
    with st.form("keys_form"):
        for origin in origins_in(reg):
            st.subheader(origin)
            for n in [n for n, p in reg.items() if p["origin"] == origin]:
                if reg[n].get("nokey"):
                    st.caption(f"{n}: no key required")
                    continue
                st.text_input(f"{n}   ({reg[n]['env']})", type="password", key=f"kf_{n}", value=keys.get(n, ""))
        if st.form_submit_button("Save keys", type="primary"):
            for n in reg:
                if f"kf_{n}" in st.session_state:
                    keys[n] = st.session_state[f"kf_{n}"].strip()
            st.success("Keys saved for this session")
            st.rerun()
    st.subheader("Status")
    show_df(pd.DataFrame([{"platform": n, "origin": reg[n]["origin"], "key": key_status(reg, n)} for n in reg]))
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Test a platform", list(reg), key="key_test_pick")
    if c2.button("Test connection", key="key_test_btn"):
        try:
            text, _, _ = llm_raw(pick, prov(pick)["model"], "Reply with the single word OK.", "Connection check.", 20)
            st.success(f"Connected. Reply: {text[:80]}")
        except Exception as e:
            st.error(str(e))
    st.subheader("Add another AI platform")
    st.caption("Works with any service that offers an OpenAI-compatible chat completions API.")
    with st.form("custom_provider", clear_on_submit=True):
        name = st.text_input("Platform name")
        base = st.text_input("Base URL", placeholder="https://api.example.com/v1")
        model = st.text_input("Default model")
        origin = st.text_input("Country or group", value="Custom")
        a, b = st.columns(2)
        pin = a.number_input("USD per 1M input tokens", min_value=0.0, value=0.0)
        pout = b.number_input("USD per 1M output tokens", min_value=0.0, value=0.0)
        nokey = st.checkbox("No API key required (local server)")
        if st.form_submit_button("Add platform"):
            if not (name.strip() and base.strip() and model.strip()):
                st.error("Name, base URL and model are required.")
            elif name.strip() in reg:
                st.error("A platform with that name already exists.")
            else:
                ex("insert into providers values(?,?,?,?,?)", (name.strip(), base.strip(), model.strip(), pin, pout))
                ex("insert into custom_providers values(?,?,?)", (name.strip(), origin.strip() or "Custom", int(nokey)))
                st.rerun()
    custom = [r["name"] for r in q("select name from custom_providers")]
    if custom:
        d1, d2 = st.columns([3, 1])
        gone = d1.selectbox("Remove a platform you added", custom, key="custom_remove")
        if d2.button("Remove", key="custom_remove_btn"):
            ex("delete from custom_providers where name=?", (gone,))
            ex("delete from providers where name=?", (gone,))
            st.rerun()


def page_providers():
    st.title("Platform settings and integrations")
    st.caption("Set the base URL, default model and prices for each platform. Add keys on the API keys page.")
    reg = registry()
    for origin in origins_in(reg):
        st.subheader(origin)
        for n in [n for n, p in reg.items() if p["origin"] == origin]:
            p = prov(n)
            with st.expander(f"{n}   -   key: {key_status(reg, n)}"):
                base = st.text_input("Base URL", p["base"], key=f"base_{n}")
                model = model_input(n, p["model"], f"model_{n}")
                if n == "OpenRouter" or "openrouter.ai" in p["base"]:
                    info = next((m for m in openrouter_models(n)[0] if m["id"] == model), None)
                    if info:
                        st.caption(f"Listed price: ${info['pin']}/1M input, ${info['pout']}/1M output, context {info['ctx']:,} tokens")
                        if st.button("Apply listed prices", key=f"orp_{n}"):
                            ex("update providers set model=?, pin=?, pout=? where name=?", (model, info["pin"], info["pout"], n))
                            st.rerun()
                a, b = st.columns(2)
                pin = a.number_input("USD per 1M input tokens", value=float(p["pin"]), min_value=0.0, key=f"pin_{n}")
                pout = b.number_input("USD per 1M output tokens", value=float(p["pout"]), min_value=0.0, key=f"pout_{n}")
                s1, t1 = st.columns(2)
                if s1.button("Save", key=f"save_{n}"):
                    ex("update providers set base=?, model=?, pin=?, pout=? where name=?", (base, model, pin, pout, n))
                    st.success("Saved")
                if t1.button("Test connection", key=f"test_{n}"):
                    try:
                        text, _, _ = llm_raw(n, model, "Reply with the single word OK.", "Connection check.", 20)
                        st.success(f"Connected. Reply: {text[:80]}")
                    except Exception as e:
                        st.error(str(e))
    st.info("Webhooks, Telegram, email, GitHub, MCP servers and other applications are configured on the Connections page.")


def provider_select(label, current, key):
    names = list(registry())
    return st.selectbox(label, names, index=names.index(current) if current in names else 0, key=key)


def page_agents():
    st.title("Agents and roles")
    st.subheader("Create a role from a description")
    desc = st.text_area("Describe the role you need", key="role_desc", placeholder="Example: an agent that reviews customer support replies for tone and accuracy")
    gen_by = st.selectbox("Generated by", [a["name"] for a in agents_by()], key="role_gen_by")
    if st.button("Generate and create role") and desc.strip():
        try:
            with st.spinner("Designing role..."):
                raw = call_llm(agent_by_name(gen_by), (
                    f"Design an AI agent role for: {desc}\nReturn only JSON: "
                    '{"name":"short unique name","role":"one sentence","system_prompt":"detailed instructions for the agent"}'), max_tokens=1500)
                r = parse_json(raw)
                base = agent_by_name(gen_by)
                ex("insert into agents(name,kind,role,system_prompt,provider,model) values(?,?,?,?,?,?)",
                   (r["name"], "general", r["role"], r["system_prompt"], base["provider"], base["model"]))
            st.success(f"Created agent {r['name']}")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    st.subheader("Manage agents")
    kinds = ["ceo", "specialist", "executor", "general"]
    for a in agents_by():
        with st.expander(f"{a['name']}  -  {a['kind']}  -  {a['provider']}"):
            name = st.text_input("Name", a["name"], key=f"an_{a['id']}")
            kind = st.selectbox("Type", kinds, index=kinds.index(a["kind"]) if a["kind"] in kinds else 3, key=f"ak_{a['id']}")
            role = st.text_input("Role", a["role"], key=f"ar_{a['id']}")
            sp = st.text_area("System prompt", a["system_prompt"], key=f"as_{a['id']}", height=120)
            provider = provider_select("Provider", a["provider"], f"ap_{a['id']}")
            model = model_input(provider, a["model"] if provider == a["provider"] else prov(provider)["model"], f"am_{a['id']}_{provider}")
            s, d = st.columns(2)
            if s.button("Save", key=f"asv_{a['id']}"):
                ex("update agents set name=?,kind=?,role=?,system_prompt=?,provider=?,model=? where id=?",
                   (name, kind, role, sp, provider, model, a["id"]))
                st.success("Saved")
                st.rerun()
            if d.button("Delete", key=f"adl_{a['id']}"):
                ex("delete from agents where id=?", (a["id"],))
                st.rerun()


def page_tasks():
    st.title("Tasks")
    agents = agents_by()
    st.subheader("Generate tasks from an objective")
    obj = st.text_area("Objective", key="task_obj")
    planner = st.selectbox("Planning agent", [a["name"] for a in agents], key="task_planner")
    if st.button("Generate tasks") and obj.strip():
        try:
            with st.spinner("Planning..."):
                names = ", ".join(a["name"] for a in agents)
                raw = call_llm(agent_by_name(planner), (
                    f"Objective: {obj}\nAvailable agents: {names}\nReturn only a JSON array: "
                    '[{"title":"short title","description":"clear instructions","agent":"exact agent name"}]'), max_tokens=2000)
                for t in parse_json(raw, "array"):
                    ag = agent_by_name(t.get("agent", "")) or agents[0]
                    ex("insert into tasks(title,description,agent_id) values(?,?,?)", (t["title"], t["description"], ag["id"]))
            st.rerun()
        except Exception as e:
            st.error(str(e))
    st.subheader("Create a task manually")
    with st.form("new_task", clear_on_submit=True):
        title = st.text_input("Title")
        description = st.text_area("Description")
        agent = st.selectbox("Assign to", [a["name"] for a in agents])
        if st.form_submit_button("Add task") and title:
            ex("insert into tasks(title,description,agent_id) values(?,?,?)", (title, description, agent_by_name(agent)["id"]))
            st.rerun()
    st.subheader("Task board")
    for t in q("select t.*, a.name agent from tasks t left join agents a on a.id=t.agent_id order by t.id desc"):
        with st.expander(f"[{t['status']}] {t['title']}  -  {t['agent']}"):
            st.write(t["description"])
            if t["result"]:
                st.markdown(t["result"])
            c1, c2 = st.columns(2)
            if t["status"] != "done" and c1.button("Run task", key=f"rt_{t['id']}"):
                try:
                    with st.spinner("Agent working..."):
                        ag = dict(q("select * from agents where id=?", (t["agent_id"],))[0])
                        res = call_llm(ag, t["description"] or t["title"])
                    ex("update tasks set status='done', result=?, finished=current_timestamp where id=?", (res, t["id"]))
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if c2.button("Delete", key=f"dt_{t['id']}"):
                ex("delete from tasks where id=?", (t["id"],))
                st.rerun()


def show_df(df):
    """st.dataframe with full width on every Streamlit version (width='stretch' only exists in new releases)."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def is_cloud_tag(m):
    return str(m).endswith("-cloud") or str(m).endswith(":cloud")


@st.cache_data(ttl=600, show_spinner=False)
def _openrouter_models(base):
    r = requests.get(base.rstrip("/") + "/models", timeout=15)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        pr = m.get("pricing") or {}
        try:
            pin, pout = float(pr.get("prompt", 0)) * 1e6, float(pr.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            pin = pout = 0.0
        out.append(dict(id=m["id"], pin=round(pin, 4), pout=round(pout, 4), ctx=m.get("context_length") or 0))
    return sorted(out, key=lambda x: x["id"])


def openrouter_models(name):
    """(models with listed prices, error text). The list is public, no key needed. Never raises."""
    try:
        return _openrouter_models(prov(name)["base"]), ""
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def model_input(provider, current, key):
    """Picker of real models for Ollama and OpenRouter, plain text box for every other platform."""
    if registry()[provider]["kind"] == "ollama":
        names, _ = ollama_models(provider)
        if names:
            opts = names + ([current] if current and current not in names else [])
            return st.selectbox("Model (available in Ollama)", opts, index=opts.index(current) if current in opts else 0, key=key,
                                format_func=lambda m: m + ("   [online]" if is_cloud_tag(m) else ""))
    elif provider == "OpenRouter" or "openrouter.ai" in prov(provider)["base"]:
        models, err = openrouter_models(provider)
        if models:
            free = st.checkbox("Free models only", key=f"{key}_free")
            ids = [m["id"] for m in models if not free or m["id"].endswith(":free") or (m["pin"] == 0 and m["pout"] == 0)]
            opts = ids + ([current] if current and current not in ids else [])
            return st.selectbox(f"Model ({len(ids)} on OpenRouter - type to search)", opts,
                                index=opts.index(current) if current in opts else 0, key=key)
        if err:
            st.caption("Could not load the OpenRouter model list (" + err[:90] + "). Type a model id instead.")
    return st.text_input("Model", current, key=key)


ONLINE = ["OpenRouter", "Ollama Cloud (online API)"]
CLOUD_SUGGEST = "gpt-oss:120b-cloud, gpt-oss:20b-cloud, qwen3.5:cloud, deepseek-v4-flash:cloud"


def online_quick_setup():
    st.subheader("Online models - quick setup")
    st.caption("OpenRouter gives one key for hundreds of models (many are free). Ollama Cloud runs large open models on ollama.com. "
               "Keys stay in this browser session; to keep them across restarts set the environment variable shown.")
    reg = registry()
    keys = st.session_state.setdefault("keys", {})
    name = st.radio("Platform", ONLINE, horizontal=True, key="oq_prov")
    link = "https://openrouter.ai/keys" if name == "OpenRouter" else "https://ollama.com/settings/keys"
    key = st.text_input(f"{name} API key   ({reg[name]['env']})   -   create one at {link}", type="password",
                        key=f"oq_key_{name}", value=keys.get(name, ""))
    model = model_input(name, prov(name)["model"], f"oq_model_{name}")

    def save():
        if key.strip():
            keys[name] = key.strip()
        ex("update providers set model=? where name=?", (model, name))
        if name == "OpenRouter":  # keep the wallet accurate: apply the model's listed price
            info = next((m for m in openrouter_models(name)[0] if m["id"] == model), None)
            if info:
                ex("update providers set pin=?, pout=? where name=?", (info["pin"], info["pout"], name))

    a, b = st.columns(2)
    if a.button("Save and test", key="oq_test"):
        save()
        try:
            text, _, _ = llm_raw(name, model, "Reply with the single word OK.", "Connection check.", 20)
            st.success(f"Connected. Reply: {text[:80]}")
        except Exception as e:
            st.error(str(e))
    if b.button("Save and use for all agents", key="oq_all", type="primary"):
        save()
        ex("update agents set provider=?, model=?", (name, model))
        st.success(f"All agents now use {name} / {model}")
    st.divider()


def ollama_panel():
    models, err = ollama_models()
    with st.expander("Ollama (local and online models)", expanded=not models):
        if err:
            st.error(f"Ollama is not reachable ({err}).")
            st.code("ollama serve\nollama pull llama3.2", language="bash")
            st.caption("Start Ollama, or set the address on the Platform settings page (default http://localhost:11434, or the OLLAMA_HOST environment variable).")
            return
        online = [m for m in models if is_cloud_tag(m)]
        st.success(f"Ollama is running: {len(models) - len(online)} local model(s), {len(online)} online model(s).")
        ags = agents_by()
        off = [a["name"] for a in ags if a["provider"] != OLLAMA]
        missing = [a["name"] for a in ags if a["provider"] == OLLAMA and a["model"] not in models]
        if off:
            st.info("Agents not using this Ollama: " + ", ".join(off))
        if missing:
            st.warning("These agents use an Ollama model that is not available: " + ", ".join(missing))
        if models:
            choice = st.selectbox("Model for all agents", models, key="oll_choice",
                                  format_func=lambda m: m + ("   [online]" if is_cloud_tag(m) else ""))
            a, b = st.columns(2)
            if a.button("Use this model for all agents", key="oll_all"):
                ex("update agents set provider=?, model=?", (OLLAMA, choice))
                st.rerun()
            if b.button("Test this model", key="oll_test"):
                try:
                    text, _, _ = llm_raw(OLLAMA, choice, "Reply with the single word OK.", "Connection check.", 20)
                    st.success(f"Model replied: {text[:60]}")
                except Exception as e:
                    st.error(str(e))
        st.markdown("**Add an online model through your installed Ollama**")
        st.caption("Online models run on Ollama's servers but you use them through your local Ollama. One-time setup: run `ollama signin` "
                   f"in a terminal. Examples: {CLOUD_SUGGEST}. The current list is at ollama.com/search?c=cloud (models are retired from time to time). "
                   "Prefer no terminal? Use Ollama Cloud with an API key on the API keys page.")
        c1, c2 = st.columns([3, 1])
        cname = c1.text_input("Online model name", "gpt-oss:120b-cloud", key="oll_cloud_name")
        use_all = c1.checkbox("Use it for all agents", value=True, key="oll_cloud_all")
        if c2.button("Add online model", key="oll_cloud_add"):
            try:
                r = requests.post(ollama_host(prov(OLLAMA)["base"]) + "/api/pull", json={"model": cname.strip(), "stream": False}, timeout=600)
                if r.status_code in (401, 403):
                    raise RuntimeError("Not signed in to Ollama. Run 'ollama signin' in a terminal, then try again.")
                _check(r, "Ollama")
                _ollama_tags.clear()
                if use_all:
                    ex("update agents set provider=?, model=?", (OLLAMA, cname.strip()))
                st.success(f"{cname.strip()} is available" + (" and all agents now use it." if use_all else "."))
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach Ollama. Is it running?")
            except Exception as e:
                st.error(str(e))
        st.caption("Agents call tools by replying in JSON, so pick an instruction-tuned model (local: about 7B or larger, e.g. llama3.1/3.2, qwen2.5, mistral-nemo). "
                   "Context size is set on the Connections page.")


def render_run(plan, outputs, summary, review):
    st.subheader("CEO plan")
    st.json(plan)
    st.subheader("Specialist output")
    for o in outputs:
        st.markdown(o)
    st.subheader("Executor report")
    st.markdown(summary)
    st.subheader("CEO review")
    st.json(review)
    if review.get("note"):
        st.success(f"Saved to the vault as {review['note']}")
    pend = q("select count(*) c from actions where status='pending'")[0]["c"]
    if pend:
        st.warning(f"{pend} action(s) are waiting for your approval on the Actions page.")


def do_run(goal):
    try:
        with st.status("Running...", expanded=True) as status:
            res = run_orchestra(goal, lambda m: st.write(m))
            status.update(label="Completed", state="complete")
        render_run(*res)
    except Exception as e:
        st.error(str(e))


def page_orchestra():
    st.title("Orchestra")
    st.caption("The CEO plans and sets a measurable target, specialists work (the Researcher can search the web and your vault), "
               "the Executor acts through tools and connections, and the CEO reviews and records the run in your vault.")
    for a in agents_by():
        st.write(f"**{a['name']}** ({a['kind']}) - {a['provider']} / {a['model']}")
    goal = st.text_area("Goal for the organisation", height=120)
    if st.button("Run orchestra", type="primary") and goal.strip():
        do_run(goal)
    st.subheader("Workspace files produced by the agents")
    for f in [f for f in sorted(WORK.rglob("*")) if f.is_file() and not f.name.startswith(".")]:
        with st.expander(str(f.relative_to(WORK))):
            st.code(f.read_text(encoding="utf-8", errors="replace")[:8000])
            st.download_button("Download", f.read_bytes(), file_name=f.name, key=f"dl_{f}")


def page_playbooks():
    st.title("Playbooks")
    st.caption("Ready-made workflows for real work. Fill in the blanks and the whole organisation runs it: research, write, act through your connections, and remember the result in the vault.")
    rows = q("select * from playbooks order by name")
    if not rows:
        st.info("No playbooks yet.")
    else:
        pick = st.selectbox("Playbook", [r["name"] for r in rows])
        pb = next(r for r in rows if r["name"] == pick)
        st.write(pb["description"])
        fields = list(dict.fromkeys(re.findall(r"\{(\w+)\}", pb["template"])))
        vals = {f: st.text_input(f.replace("_", " ").capitalize(), key=f"pbv_{pick}_{f}") for f in fields}
        with st.expander("Instructions the CEO receives"):
            st.code(pb["template"])
        a, b = st.columns([1, 1])
        if a.button("Run playbook", type="primary"):
            missing = [f for f, v in vals.items() if not v.strip()]
            if missing:
                st.error("Please fill in: " + ", ".join(missing))
            else:
                do_run(re.sub(r"\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), pb["template"]))
        if b.button("Delete this playbook", key="pb_del"):
            ex("delete from playbooks where name=?", (pick,))
            st.rerun()
    st.subheader("Add your own playbook")
    with st.form("new_playbook", clear_on_submit=True):
        name = st.text_input("Name")
        desc = st.text_input("Description")
        tpl = st.text_area("Instructions (use {placeholders} for the values you will fill in)", height=140)
        if st.form_submit_button("Save playbook") and name.strip() and tpl.strip():
            ex("insert or replace into playbooks values(?,?,?)", (name.strip(), desc.strip(), tpl.strip()))
            st.rerun()


PH = {"url": "https://hooks.slack.com/services/...", "chat_id": "123456789", "host": "smtp.gmail.com", "port": "587",
      "user": "you@example.com", "from_addr": "you@example.com", "default_to": "you@example.com",
      "repo": "owner/name", "auth_header": "Authorization: Bearer", "default_headers": '{"Accept":"application/json"}'}
PH_BASE = {"github": "https://api.github.com   (Gitea/Forgejo: https://host/api/v1)", "rest": "http://localhost:8123/api",
           "searxng": "http://localhost:8080"}


def page_connections():
    st.title("Connections and runtime")
    st.subheader("Safety and runtime")
    with st.form("runtime"):
        a, b = st.columns(2)
        mode = a.radio("Risky actions (send message or email, create issue, REST write, MCP call, run code)",
                       ["ask", "auto"], index=0 if setting("approval_mode", "ask") == "ask" else 1,
                       format_func=lambda x: "Ask me first (queue on Actions page)" if x == "ask" else "Run automatically")
        code = b.checkbox("Allow agents to run Python code in the workspace (not a sandbox: only enable on a machine you trust)",
                          value=setting("allow_code", "0") == "1")
        c, d, e = st.columns(3)
        steps = c.number_input("Max tool steps per agent", 3, 40, int(setting("max_steps", "10")))
        ctx = d.number_input("Ollama context size (tokens)", 2048, 131072, int(setting("ollama_ctx", "8192")), step=2048)
        temp = e.number_input("Ollama temperature", 0.0, 1.5, float(setting("ollama_temp", "0.3")), step=0.1)
        vpath = st.text_input("Vault folder (point this at your Obsidian vault, or leave empty for data/vault)", setting("vault_path"))
        if st.form_submit_button("Save runtime settings", type="primary"):
            for k, v in dict(approval_mode=mode, allow_code="1" if code else "0", max_steps=str(int(steps)), ollama_ctx=str(int(ctx)),
                             ollama_temp=str(temp), vault_path=vpath.strip()).items():
                set_setting(k, v)
            st.success("Saved")
            st.rerun()

    st.subheader("Your connections")
    st.caption("Secrets (tokens, passwords) stay in this browser session or in an environment variable named CONN_<NAME>_SECRET. Only non-secret settings are stored in the database.")
    types = cx.CONNECTION_TYPES
    secrets = st.session_state.setdefault("conn_secrets", {})
    for r in q("select * from connections order by name"):
        cfg = json.loads(r["config"] or "{}")
        t = types.get(r["type"], dict(label=r["type"], fields=list(cfg), secret=False))
        envname = "CONN_" + re.sub(r"[^A-Z0-9]", "_", r["name"].upper()) + "_SECRET"
        with st.expander(f"{r['name']}  -  {t['label']}"):
            new = {f: st.text_input(f, cfg.get(f, ""), key=f"cf_{r['name']}_{f}") for f in t["fields"]}
            sec = ""
            if t["secret"]:
                sec = st.text_input(f"Secret / token   ({envname})", type="password", key=f"cs_{r['name']}",
                                    placeholder="set" if conn_secret(r["name"]) else "missing")
            b1, b2, b3 = st.columns(3)
            if b1.button("Save", key=f"cv_{r['name']}"):
                ex("update connections set config=? where name=?", (json.dumps(new), r["name"]))
                if sec.strip():
                    secrets[r["name"]] = sec.strip()
                st.success("Saved")
            if b2.button("Test", key=f"ct_{r['name']}"):
                if sec.strip():
                    secrets[r["name"]] = sec.strip()
                sync_ctx()
                ok, msg = cx.test_connection(dict(name=r["name"], type=r["type"], config=new))
                (st.success if ok else st.error)(msg)
            if b3.button("Delete", key=f"cd_{r['name']}"):
                ex("delete from connections where name=?", (r["name"],))
                st.rerun()
            if r["type"] == "webhook" and st.button("Send test message", key=f"cw_{r['name']}"):
                sync_ctx()
                st.info(cx.call_tool("send_webhook", {"text": "Test message from AI Agent Orchestra", "connection": r["name"]}))
    st.subheader("Add a connection")
    ctype = st.selectbox("Type", list(types), format_func=lambda k: types[k]["label"], key="new_conn_type")
    name = st.text_input("Connection name (the agents see this name)", key="new_conn_name")
    vals = {f: st.text_input(f, key=f"nc_{ctype}_{f}", placeholder=PH_BASE.get(ctype, "") if f in ("base",) else PH.get(f, ""))
            for f in types[ctype]["fields"]}
    sec = st.text_input("Secret / token", type="password", key=f"nc_{ctype}_secret") if types[ctype]["secret"] else ""
    if st.button("Add connection", key="add_conn"):
        if not name.strip():
            st.error("Give the connection a name.")
        elif q("select 1 from connections where name=?", (name.strip(),)):
            st.error("A connection with that name already exists.")
        else:
            ex("insert into connections values(?,?,?)", (name.strip(), ctype, json.dumps(vals)))
            if sec.strip():
                secrets[name.strip()] = sec.strip()
            st.rerun()
    st.caption("MCP servers open the door to hundreds of open-source tools (filesystem, databases, Git, browsers, Obsidian, Home Assistant ...). "
               "Add one here, then the Executor can use mcp_list_tools and mcp_call. Only HTTP (streamable) MCP servers are supported.")

    st.subheader("Tool tester")
    st.caption("Run any tool yourself to check a connection works. Runs immediately; the approval queue only applies to agents.")
    tool = st.selectbox("Tool", list(cx.TOOLS), key="tt_tool")
    args_txt = st.text_area("Arguments (JSON)", cx.TOOLS[tool][3].split("  or")[0], key=f"tt_args_{tool}", height=90)
    if st.button("Run tool", key="tt_run"):
        try:
            sync_ctx()
            st.code(cx.call_tool(tool, json.loads(args_txt)))
        except Exception as e:
            st.error(str(e))


def page_actions():
    st.title("Actions")
    st.caption("Everything the agents did through connections, plus anything waiting for your approval.")
    if st.session_state.get("act_msg"):
        st.info("Last approved action result: " + st.session_state.pop("act_msg"))
    pend = q("select * from actions where status='pending' order by id")
    st.subheader(f"Waiting for approval ({len(pend)})")
    if not pend:
        st.info("Nothing is waiting.")
    for a in pend:
        with st.expander(f"#{a['id']}  {a['tool']}  -  requested by {a['agent']}", expanded=True):
            st.code(json.dumps(json.loads(a["args"] or "{}"), indent=2, ensure_ascii=False), language="json")
            c1, c2 = st.columns(2)
            if c1.button("Approve and run", key=f"ap_{a['id']}", type="primary"):
                st.session_state["act_msg"] = approve_action(a["id"])[:500]
                st.rerun()
            if c2.button("Reject", key=f"rj_{a['id']}"):
                ex("update actions set status='rejected' where id=?", (a["id"],))
                st.rerun()
    st.subheader("History")
    rows = q("select id, created, agent, tool, status, substr(result,1,200) result from actions where status!='pending' order by id desc limit 100")
    if rows:
        show_df(pd.DataFrame([dict(r) for r in rows]))


def page_builder():
    st.title("App builder")
    agents = agents_by()
    default = next((i for i, a in enumerate(agents) if a["name"] == "Developer"), 0)
    agent = st.selectbox("Builder agent", [a["name"] for a in agents], index=default)
    name = st.text_input("App file name", "my_app")
    spec = st.text_area("What should the app do?", height=150)
    if st.button("Build app") and spec.strip():
        try:
            with st.spinner("Building..."):
                raw = call_llm(agent_by_name(agent), (
                    f"Write a complete, working single-file Streamlit app that does the following:\n{spec}\n\n"
                    "Use only widely available Python packages, handle errors, and return only the code in one ```python block."),
                    max_tokens=6000)
            m = re.search(r"```(?:python)?\n(.*?)```", raw, re.S)
            code = m.group(1) if m else raw
            fname = re.sub(r"[^a-zA-Z0-9_]", "_", name) + ".py"
            path = safe_path(f"apps/{fname}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
            st.session_state["built"] = (fname, code)
        except Exception as e:
            st.error(str(e))
    if "built" in st.session_state:
        fname, code = st.session_state["built"]
        st.success(f"Saved to data/workspace/apps/{fname}. Run it with: streamlit run data/workspace/apps/{fname}")
        st.code(code, language="python")
        st.download_button("Download app", code, file_name=fname)


def page_wallet():
    st.title("Wallet")
    st.caption("Each agent has a USD budget. Spend is calculated from real token usage reported by the providers and the prices set on the Providers page. An agent stops when its wallet is empty.")
    rows = agents_by()
    df = pd.DataFrame([{"agent": a["name"], "provider": a["provider"], "budget": a["budget_usd"], "spent": round(a["spent_usd"], 6),
                        "remaining": round(a["budget_usd"] - a["spent_usd"], 6)} for a in rows])
    show_df(df)
    st.subheader("Allocate budget")
    c1, c2, c3 = st.columns(3)
    who = c1.selectbox("Agent", [a["name"] for a in rows])
    amount = c2.number_input("New total budget (USD)", min_value=0.0, value=5.0, step=1.0)
    if c3.button("Allocate"):
        ex("update agents set budget_usd=? where name=?", (amount, who))
        st.rerun()
    if st.button("Reset spend counters"):
        ex("update agents set spent_usd=0")
        st.rerun()
    u = q("select provider, sum(cost) cost from usage group by provider")
    if u:
        st.subheader("Spend by provider")
        st.bar_chart(pd.DataFrame([dict(r) for r in u]).set_index("provider"))
    st.subheader("Usage log")
    log = q("select u.ts, a.name agent, u.provider, u.model, u.tokens_in, u.tokens_out, round(u.cost,6) cost from usage u left join agents a on a.id=u.agent_id order by u.id desc limit 100")
    if log:
        show_df(pd.DataFrame([dict(r) for r in log]))


def render_note(text):
    meta, body = vaultlib.parse(text)
    st.markdown(vaultlib.WIKI.sub(lambda m: f"**{m.group(1).strip()}**", body))


def page_memory():
    st.title("Memory (Obsidian-style vault)")
    v = vault()
    notes = v.notes()
    st.caption(f"Vault folder: {v.root}  -  {len(notes)} notes. Open this folder in Obsidian (Open folder as vault), or point the app at your "
               "existing vault on the Connections page. Agents read the most relevant notes before every call and write notes with [[links]] and #tags.")
    tags = v.all_tags()
    if tags:
        st.caption("Tags: " + "  ".join(f"#{t} ({n})" for t, n in tags.most_common(12)))
    tabs = st.tabs(["Browse and edit", "Search", "Graph", "New note", "Facts"])
    with tabs[0]:
        rels = [v.rel(p) for p in notes]
        if not rels:
            st.info("The vault is empty. Run the Orchestra or a playbook, or create a note on the New note tab.")
        else:
            pick = st.selectbox("Note", rels, key="vault_pick")
            txt = v.read(pick)
            left, right = st.columns([3, 1])
            with left:
                render_note(txt)
            with right:
                st.markdown("**Links**")
                for n in v.links(pick):
                    st.caption(n)
                st.markdown("**Backlinks**")
                for n in v.backlinks(pick):
                    st.caption(n)
            with st.expander("Edit this note"):
                edited = st.text_area("Markdown", txt, height=300, key=f"vault_edit_{pick}")
                c1, c2 = st.columns(2)
                if c1.button("Save note", key=f"vault_save_{pick}"):
                    v.write(pick, edited)
                    st.success("Saved")
                if c2.button("Delete note", key=f"vault_del_{pick}"):
                    v.delete(pick)
                    st.rerun()
    with tabs[1]:
        query = st.text_input("Search the vault", key="vault_query")
        if query.strip():
            hits = v.search(query, k=10)
            if not hits:
                st.info("No matching notes.")
            for rel, snip in hits:
                with st.expander(rel):
                    st.caption(snip)
                    render_note(v.read(rel))
    with tabs[2]:
        nodes, edges = v.graph()
        if not nodes:
            st.info("No notes yet.")
        else:
            keys = list(nodes)[:150]
            ids = {k: f"n{i}" for i, k in enumerate(keys)}
            dot = ['digraph G {rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#0b2559", '
                   'color="#38bdf8", fontcolor="#e8eefc", fontsize=10]; edge [color="#6ea8ff"];']
            for k in keys:
                dot.append(f'{ids[k]} [label="{nodes[k].replace(chr(34), "")}"];')
            for a, b in edges:
                if a in ids and b in ids:
                    dot.append(f"{ids[a]} -> {ids[b]};")
            st.graphviz_chart("\n".join(dot) + "}")
            st.caption(f"{len(nodes)} notes, {len(edges)} links" + (" (showing the first 150 notes)" if len(nodes) > 150 else ""))
    with tabs[3]:
        with st.form("new_note", clear_on_submit=True):
            path = st.text_input("Path / title", placeholder="Projects/My idea")
            ntags = st.text_input("Tags (comma separated)")
            content = st.text_area("Content (markdown, [[wikilinks]] and #tags work)", height=200)
            if st.form_submit_button("Create note") and path.strip():
                v.write(path.strip(), content, tags=[t.strip() for t in ntags.split(",") if t.strip()])
                st.rerun()
    with tabs[4]:
        st.caption("Short facts. Agents read their own facts plus shared facts (scope: all agents) before every call.")
        agents = agents_by()
        with st.form("mem", clear_on_submit=True):
            scope = st.selectbox("Scope", ["All agents"] + [a["name"] for a in agents])
            content = st.text_area("Fact")
            if st.form_submit_button("Save fact") and content.strip():
                aid = 0 if scope == "All agents" else agent_by_name(scope)["id"]
                ex("insert into memory(agent_id,content) values(?,?)", (aid, content.strip()))
                st.rerun()
        for m in q("select m.*, coalesce(a.name,'All agents') owner from memory m left join agents a on a.id=m.agent_id order by m.id desc"):
            c1, c2 = st.columns([8, 1])
            c1.write(f"**{m['owner']}**: {m['content']}")
            if c2.button("Delete", key=f"dm_{m['id']}"):
                ex("delete from memory where id=?", (m["id"],))
                st.rerun()


def token_source(name):
    reg = registry()[name]
    if reg.get("nokey"):
        return "not needed"
    if st.session_state.setdefault("keys", {}).get(name):
        return "session"
    return "environment" if os.environ.get(reg["env"]) else "missing"


def mask(v):
    return ("*" * 8 + v[-4:]) if len(v) > 8 else ("set" if v else "")


def tokens_today():
    return q("select coalesce(sum(tokens_in+tokens_out),0) t from usage where date(ts)=date('now')")[0]["t"]


def page_tokens():
    st.title("Tokens")
    st.caption("Everything about tokens in one place: your API tokens (keys) for AI platforms and connected apps, "
               "and how many AI tokens the agents have used. Tokens are kept in this browser session or in environment variables, never written to the database.")
    t1, t2, t3 = st.tabs(["API tokens", "Usage", "Limits"])
    reg = registry()
    keys = st.session_state.setdefault("keys", {})
    secrets = st.session_state.setdefault("conn_secrets", {})
    conns = [dict(r) for r in q("select * from connections order by name")]
    with t1:
        rows = [dict(Target=n, Kind="AI platform", Variable=r["env"], Source=token_source(n), Token=mask(get_key(n)))
                for n, r in reg.items()]
        for c in conns:
            if cx.CONNECTION_TYPES.get(c["type"], {}).get("secret"):
                env = "CONN_" + re.sub(r"[^A-Z0-9]", "_", c["name"].upper()) + "_SECRET"
                v = conn_secret(c["name"])
                rows.append(dict(Target=c["name"], Kind=f"Connection ({c['type']})", Variable=env,
                                 Source="session" if secrets.get(c["name"]) else ("environment" if v else "missing"), Token=mask(v)))
        show_df(pd.DataFrame(rows))
        st.subheader("Add, replace or clear a token")
        targets = [n for n, r in reg.items() if not r.get("nokey")] + [c["name"] for c in conns
                                                                       if cx.CONNECTION_TYPES.get(c["type"], {}).get("secret")]
        if targets:
            target = st.selectbox("Token for", targets, key="tk_target")
            val = st.text_input("Token", type="password", key=f"tk_val_{target}")
            a, b, c = st.columns(3)
            store = keys if target in reg else secrets
            if a.button("Save token", key="tk_save", type="primary") and val.strip():
                store[target] = val.strip()
                st.success("Saved for this session")
            if b.button("Clear token", key="tk_clear"):
                store.pop(target, None)
                st.rerun()
            if c.button("Test token", key="tk_test"):
                if val.strip():
                    store[target] = val.strip()
                if target in reg:
                    try:
                        text, _, _ = llm_raw(target, prov(target)["model"], "Reply with the single word OK.", "Connection check.", 20)
                        st.success(f"Token works. Reply: {text[:60]}")
                    except Exception as e:
                        st.error(str(e))
                else:
                    sync_ctx()
                    row = next(x for x in conns if x["name"] == target)
                    ok, msg = cx.test_connection(dict(name=row["name"], type=row["type"], config=json.loads(row["config"] or "{}")))
                    (st.success if ok else st.error)(msg)
        st.caption("To keep tokens across restarts, set the variable shown in the table before starting the app.")
    with t2:
        tot = q("select coalesce(sum(tokens_in),0) i, coalesce(sum(tokens_out),0) o, coalesce(sum(cost),0) c, count(*) n from usage")[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input tokens", f"{tot['i']:,}")
        m2.metric("Output tokens", f"{tot['o']:,}")
        m3.metric("AI calls", f"{tot['n']:,}")
        m4.metric("Estimated cost (USD)", f"{tot['c']:.4f}")
        st.subheader("By platform and model")
        rows = q("select provider, model, count(*) calls, sum(tokens_in) tokens_in, sum(tokens_out) tokens_out, round(sum(cost),4) cost "
                 "from usage group by provider, model order by sum(tokens_in+tokens_out) desc")
        if rows:
            show_df(pd.DataFrame([dict(r) for r in rows]))
        st.subheader("By agent")
        rows = q("select coalesce(a.name,'?') agent, count(*) calls, sum(u.tokens_in) tokens_in, sum(u.tokens_out) tokens_out, round(sum(u.cost),4) cost "
                 "from usage u left join agents a on a.id=u.agent_id group by u.agent_id order by sum(u.tokens_in+u.tokens_out) desc")
        if rows:
            show_df(pd.DataFrame([dict(r) for r in rows]))
        days = q("select date(ts) day, sum(tokens_in+tokens_out) tokens from usage group by date(ts) order by day desc limit 14")
        if days:
            st.subheader("Tokens per day (last 14 days)")
            st.bar_chart(pd.DataFrame([dict(r) for r in days]).set_index("day").sort_index())
        if not rows:
            st.info("No AI calls yet. Run the Orchestra or a playbook and usage will appear here.")
    with t3:
        st.caption("A daily limit stops all agents once the total tokens used today (all platforms) reach the number below. "
                   "Use it to protect a paid token balance. The USD wallet budgets per agent still apply as well.")
        cap = to_int(setting("token_cap_daily", "0"))
        used = tokens_today()
        new = st.number_input("Daily token limit (0 = no limit)", 0, 1_000_000_000, cap, step=10000, key="tk_cap")
        if st.button("Save limit", key="tk_cap_save"):
            set_setting("token_cap_daily", str(int(new)))
            st.rerun()
        st.write(f"Used today: **{used:,}**" + (f" of {cap:,}" if cap else ""))
        if cap:
            st.progress(min(used / cap, 1.0))


PAGES = {"Dashboard": page_dashboard, "Orchestra": page_orchestra, "Playbooks": page_playbooks, "Memory": page_memory,
         "Connections": page_connections, "Actions": page_actions, "Agents": page_agents, "Tasks": page_tasks,
         "App builder": page_builder, "API keys": page_keys, "Tokens": page_tokens, "Platform settings": page_providers, "Wallet": page_wallet}

THEME = """<style>
.stApp{background:linear-gradient(160deg,#050d24 0%,#0a1f4d 55%,#0d2b6b 100%);color:#e8eefc}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#040a1c 0%,#0a1a42 100%);border-right:1px solid #1f3a7a}
h1,h2,h3,h4{color:#ffffff;letter-spacing:.3px}
h1{background:linear-gradient(90deg,#6ea8ff,#9be7ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
p,label,span,li,div[data-testid="stCaptionContainer"]{color:#dbe6ff}
div[data-testid="stMetric"]{background:rgba(255,255,255,.06);border:1px solid rgba(110,168,255,.35);border-radius:14px;padding:14px 16px;box-shadow:0 6px 20px rgba(0,0,0,.35)}
div[data-testid="stMetricValue"]{color:#7dd3fc}
div[data-testid="stExpander"],div[data-testid="stForm"]{background:rgba(255,255,255,.05);border:1px solid rgba(110,168,255,.28);border-radius:14px}
.stButton>button,.stDownloadButton>button,div[data-testid="stFormSubmitButton"]>button{background:linear-gradient(90deg,#2563eb,#38bdf8);color:#fff;border:0;border-radius:10px;font-weight:600;padding:.5rem 1.1rem;transition:all .15s}
.stButton>button:hover,.stDownloadButton>button:hover,div[data-testid="stFormSubmitButton"]>button:hover{filter:brightness(1.15);transform:translateY(-1px);box-shadow:0 6px 16px rgba(56,189,248,.35)}
input,textarea,[data-baseweb="select"]>div{background:rgba(5,13,36,.85)!important;color:#e8eefc!important;border-color:#2a4a9a!important;border-radius:10px!important}
div[data-testid="stDataFrame"]{border:1px solid rgba(110,168,255,.28);border-radius:12px}
div[data-testid="stProgress"]>div>div>div>div{background:linear-gradient(90deg,#2563eb,#38bdf8)}
</style>"""


def main():
    st.set_page_config(page_title="AI Agent Orchestra", layout="wide")
    st.markdown(THEME, unsafe_allow_html=True)
    init_db()
    sync_ctx()
    st.sidebar.title("Agent Orchestra")
    page = st.sidebar.radio("Navigate", list(PAGES))
    PAGES[page]()


main()
