# AI Agent Orchestra

    pip install -r requirements.txt
    ollama serve            # in another terminal (or just open the Ollama app)
    ollama pull llama3.2    # or qwen2.5:7b, mistral-nemo, gpt-oss ...
    streamlit run app.py

- Online models: OpenRouter (API keys page: key + searchable model list, free filter, price applied) and Ollama Cloud (API key),
  or cloud models through your installed Ollama (`ollama signin`, then Add online model on the Dashboard).
- Defaults to local Ollama (http://localhost:11434, or set OLLAMA_HOST). Pick the installed model on the Dashboard.
- Memory = an Obsidian-compatible vault in data/vault (set your own vault folder on the Connections page).
- Connections page: webhook (Slack/Discord/n8n...), Telegram, email, GitHub/Gitea, any REST API, MCP servers, SearXNG.
- Risky actions (send, create issue, run code) wait on the Actions page until you approve them.
- Secrets stay in the browser session, or in env vars named CONN_<NAME>_SECRET.
- Delete data/orchestra.db to reset; your vault notes are separate and untouched.
- Tokens page: status of every API token (AI platforms and connections, masked), token usage by platform/model/agent/day, and an optional daily token limit.
