# Friction log (DRAFT: edit so every entry reflects what you actually experienced)

Participants who submit friction logs can earn up to a 10% scoring bonus, and **only real entries help**.
Entries 1-3 come from building this project; add your own from the live Alexa+/AWS deploy.

### 1. MCP Python SDK v2 renamed `FastMCP`
- **Task:** write a minimal MCP server with the Python SDK.
- **Steps:** `pip install mcp` (installed 2.x), `from mcp.server.fastmcp import FastMCP`.
- **Expected:** import works as in most tutorials.
- **Actual:** `ModuleNotFoundError` pointing to `MCPServer` and a migration guide.
- **Severity:** Medium (clear message, ~5 min lost).
- **Workaround:** `from mcp.server.mcpserver import MCPServer`.
- **Suggestion:** hackathon starter docs should pin an SDK major version or show v2 snippets.

### 2. Which MCP protocol revision will Alexa+ negotiate?
- **Task:** confirm the server satisfies "spec 2025-11-25 or later".
- **Actual:** the SDK client negotiated 2026-07-28 by default; the rules name 2025-11-25 as the minimum.
- **Severity:** Low-Medium (uncertainty, not a failure).
- **Workaround:** verified `initialize` with `protocolVersion: 2025-11-25` is answered with 2025-11-25 and tool calls work (automated in `tests/test_mcp_e2e.py`).
- **Suggestion:** state exactly which revision(s) Alexa+ sends.

### 3. How do I get Alexa+ to call a tool I wasn't explicitly asked for?
- **Task:** have Alexa+ call `report_fact` whenever the user mentions relevant news, without being asked.
- **Actual:** the Alexa+ MCP docs say its AI reasoning decides when and how to invoke tools from customer intent, and that sessions follow the customer's conversation history. I found no way to mark a tool as "call proactively". ✏️ *(confirm with your live testing and rewrite with what you observed)*
- **Severity:** Medium (core to the product's behaviour).
- **Workaround:** explicit, imperative wording in the server `instructions` and the tool description; the demo also uses explicit phrasing.
- **Suggestion:** document how server `instructions` and tool descriptions influence routing, and offer an "ambient / proactive" hint.

### 4. Per-user state needs account linking, and prototype requirements are unclear
- **Task:** keep separate memory per family member.
- **Actual:** the docs say personalised results are available for account-linked customers, and that there is no explicit session identifier; the QuickStart also lists OpenID Connect. It was unclear what a single-household prototype may skip. ✏️ *(verify against the docs; keep or delete)*
- **Severity:** Medium (affects architecture).
- **Workaround:** one household per deployment, plus an optional bearer token.
- **Suggestion:** a "prototype mode" section with the minimum auth/identity needed to demo end to end.

### 5+. ✏️ Your live-deploy findings (Bedrock model access/region, `alexa-ai` CLI, HTTPS hosting…)
