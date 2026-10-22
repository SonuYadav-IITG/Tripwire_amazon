# 🪤 Tripwire: a voice memory for decisions *and what would change them*

> "Alexa, we're going with the 200 meg internet plan because it's cheapest. We'd switch if the price goes above $50
> or outages hit more than twice a month."
> … three weeks later …
> "Alexa, the provider just raised our price to $55."  →  **"That trips your internet plan decision. Keep it, change it, or adjust the limit?"**

Built for the **Amazon Developer Hackathon 2026**: **Alexa+ track** + **AWS Builder mini challenge**.

Assistants remember **facts**. Tripwire remembers **why you decided and what would reverse it**, and quietly
checks every new thing you mention against those conditions. It is a **self-hosted MCP server** (Streamable HTTP,
MCP spec **2025-11-25**, also serves later revisions) that Alexa+ can call as an add-on, with **Amazon Bedrock**
for language understanding and **Amazon DynamoDB** for long-lived memory.

> **Credit:** the *tripwire* idea (decide in advance what would make you reconsider) comes from Chip and Dan Heath's book
> *Decisive*, and "kill criteria" are standard practice in product and investing circles. Tripwire's contribution is making that
> technique **voice-native and automatic**: conditions are captured by speaking, and checked against whatever you mention later.
> Existing Alexa reminders are time-based; Tripwire is condition-based.

## Why this isn't "a single-turn Q&A bot"

| Capability | How |
|---|---|
| **Memory that spans conversations and weeks** | Each decision, its tripwires and its full history live in DynamoDB. A new Alexa conversation can fire a tripwire set in an old one. |
| **Built to be ambient** | Tool descriptions and server `instructions` tell the assistant to call `report_fact` whenever you mention relevant news. **Caveat:** Amazon's docs say Alexa+ decides *when* to invoke tools from your intent, so how reliably it does this unprompted must be verified on a real device. Saying "tell Tripwire…" always works. |
| **LLM proposes, server disposes** | Bedrock (Converse + *forced tool use*) extracts structured tripwires and relates facts to them, but **numeric thresholds are compared by the server**. A hallucinated "tripped" can't flip a decision; a hallucinated "not tripped" can't hide a real breach. Both are tested adversarially. |
| **Honest uncertainty** | Qualitative tripwires ("cushions go flat") are never auto-fired: they become *pending* and Alexa asks you. It warns when you're within 10% of a limit. It asks "which one?" instead of guessing between decisions or limits. |
| **A decision timeline** | "Why did we pick that plan?" replays the reasoning, the alternatives, and everything that's happened since (tripped, kept, limit moved…). |

## MCP tools

| Tool | Purpose |
|---|---|
| `record_decision` | Pass the user's words; Bedrock extracts title, choice, options, reason, tripwires, optional check-in date. Reads the tripwires back, numbered. |
| `report_fact` | Check a new fact against every armed tripwire (the ambient tool). |
| `whats_on_watch` | Live decisions; flags tripped ones and check-in dates that have arrived. |
| `why_did_we` | Reasoning + alternatives + tripwires + history for a decision. |
| `confirm_tripwire` | Human yes/no on a *possible* (qualitative) match. |
| `resolve_decision` | After a trip: `keep` (optionally move one numeric limit), `change`, or `close`. |
| `add_tripwire` / `remove_tripwire` | Edit what's being watched. |

Read-only tools carry `readOnlyHint`; `remove_tripwire` is marked destructive.

## Quick start (no AWS needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                                  # 93 tests, incl. end-to-end over real HTTP

TRIPWIRE_AI=offline tripwire &                    # serves http://127.0.0.1:8000/mcp
python scripts/demo_client.py                     # narrated demo of the whole story
```

`TRIPWIRE_AI=offline` uses a small rule-based parser (handles phrasings like *"…if the price goes above $50"*)
so you can try everything without cloud access. With `auto` (default) the server tries Bedrock and, if unavailable,
falls back **and says so** (`"engine": "offline-fallback"`). `bedrock` mode never falls back.

## Run it in Google Colab

`notebooks/tripwire_colab.ipynb` is a self-contained notebook: it writes the server to disk, starts the real MCP server inside
Colab, plays the full demo conversation with a real MCP client, and includes a "lying model" safety proof. It works with no AWS
account; optional sections add real Bedrock + DynamoDB (via Colab Secrets) and a temporary public HTTPS URL for Alexa+.

Open it with: `https://colab.research.google.com/github/<your-github-user>/tripwire/blob/main/notebooks/tripwire_colab.ipynb`
(or upload the `.ipynb` file at colab.research.google.com). Colab sessions are temporary, so use the notebook for demos and
development, and deploy the repo somewhere permanent for the final submission.

## Demo page for the video

`docs/demo/index.html` is a self-contained visual replay (decisions, tripwire gauges, the lying-model safety proof) built from
**real server output**. Open it in a browser while recording; ← → step, Space autoplays. It is a visualiser, not a live Alexa device,
so pair it with real Alexa+ footage. It can be hosted on GitHub Pages as-is.

## Reliability & failure handling

| Risk | What Tripwire does |
|---|---|
| **Slow / hung AWS call** | Bedrock: 3 s connect, 6 s read, **2 total attempts** (worst case about 12 s). DynamoDB: 2 s / 4 s / 3 attempts. All tunable via env vars (`.env.example`). |
| **Bedrock outage** | Circuit breaker: after 3 consecutive failures Bedrock is skipped for 60 s, so each voice turn doesn't pay the timeout. `auto` mode falls back to the offline parser and says so (`engine: offline-fallback`); `bedrock` mode fails fast with a spoken error. |
| **Truncated / garbage model output** | Answers with `stopReason != tool_use` are refused (a cut-off call can parse into a *partial* result). Everything is schema-validated; non-finite numbers and unknown tripwire ids are dropped. |
| **Concurrent edits** | Every write is a compare-and-swap with automatic retry-and-merge, so two requests touching one decision can't overwrite each other (tested with a forced race). Tripwires have stable ids, so a verdict can never land on the wrong tripwire after an edit. |
| **Client retries** (voice platforms retry on timeout) | `record_decision` and `add_tripwire` are idempotent; re-reporting a fact never double-trips. |
| **Database down** | Callers get a short spoken apology, never a stack trace or ARN; details go to the server log. `/healthz` = process alive, `/readyz` = database reachable (503 otherwise). |
| **Hostile / degenerate input** | Length caps, curly-quote and whitespace normalisation, fuzzed with adversarial strings under a time limit. History, tripwire and decision counts are capped. |
| **Misconfiguration** | Startup warnings for a public bind with no `TRIPWIRE_ALLOWED_HOSTS` (would 421) or no `TRIPWIRE_TOKEN`; a static test checks that the IAM policy grants every DynamoDB call the code makes. |

## Running with AWS

1. **Table + IAM:** `aws cloudformation deploy --template-file infra/template.yaml --stack-name tripwire --capabilities CAPABILITY_IAM`
   (DynamoDB on-demand, encrypted, PITR on; policy allows only `Get/Put/DeleteItem`, `Query`, `bedrock:InvokeModel`).
2. **Enable a Bedrock model** in your region (Console → Bedrock → Model access) and note its model / inference-profile ID.
3. **Configure** (see `.env.example`):
   ```bash
   export AWS_REGION=us-east-1 BEDROCK_MODEL_ID=<your model or inference profile id>
   export TRIPWIRE_STORE=dynamodb TRIPWIRE_TABLE=Tripwire TRIPWIRE_AI=bedrock TRIPWIRE_TOKEN=<long random string>
   ```
4. **Verify on your real account:** `python scripts/check_aws.py`
5. **Run:** `tripwire` (or `docker build -t tripwire . && docker run -p 8000:8000 --env-file .env tripwire`).

### AWS services used
- **Amazon Bedrock (Converse API, forced tool use, temperature 0):** `src/tripwire/ai.py`
- **Amazon DynamoDB (single table, `pk=HOUSEHOLD#id`, `sk=decision#key`):** `src/tripwire/store.py`
- **IAM + CloudFormation:** `infra/template.yaml`

## Connecting to Alexa+

Per Amazon's [Alexa+ MCP QuickStart](https://developer.amazon.com/docs/alexaplus/add-ons/mcp-toolkit-quickstart.html) the
server must speak Streamable HTTP at a remote **HTTPS** URL (reverse proxy / API gateway / tunnel; set
`TRIPWIRE_ALLOWED_HOSTS=your.domain`). Then:

```bash
alexa-ai configure
alexa-ai new mcp --name "Tripwire" --locale en-US --mcp-server-url "https://<your-host>/mcp"
alexa-ai deploy
```

Example phrases: *"Alexa, ask Tripwire to remember that we're going with…"* · *"…the provider raised our price to $55"* ·
*"…what should I look at?"* · *"…why did we pick the internet plan?"*

## Tests

`python -m pytest`: domain-logic tests (routing between decisions, limits, revisit dates, ambiguity); **moto**-backed tests
that run the real `DynamoStore`; Bedrock tests that run a real boto3 `bedrock-runtime` client under botocore's `Stubber` (every
Converse request is validated against AWS's actual API model) and feed back **adversarial model output**; and end-to-end tests
that start a real server and drive it with the real MCP client over Streamable HTTP (including protocol-version 2025-11-25
negotiation and the auth guard). See `docs/ARCHITECTURE.md` for the design rationale.

## Known limitations (honest list)

- **Single household per deployment** (`TRIPWIRE_HOUSEHOLD`). Amazon's docs say personalised results need an account-linked customer;
  per-person memory would need that OIDC/account-linking flow, which is **not implemented**. The optional bearer token protects the endpoint, not individual users.
- **Alexa+ decides when to call tools.** Ambient fact-checking depends on its intent routing (see the caveat above).
- **Tripwires fire when something is reported**, not by polling the outside world. Check-in dates surface in `whats_on_watch`; there are no proactive notifications.
- `remove_tripwire` works by number, so a blind client retry could remove a second tripwire (the other mutating tools are retry-safe).
- A tool call that exceeds the client's timeout keeps running server-side (threads can't be cancelled), so the change may still be saved; the retry-safety above covers the common cases.
- Numeric tripwires support one comparison against one number (no ranges, compound conditions or unit conversion).
- The offline parser is deliberately small (tests and no-cloud demos); real phrasing quality comes from Bedrock and depends on the model you enable. Run `scripts/check_aws.py`.
- SQLite mode is single-process (its compare-and-swap uses an in-process lock); use DynamoDB for multiple containers.
- No rate limiting or per-user quotas; put an API gateway in front for a public deployment.
- Model-generated text (tripwire descriptions) is spoken back, capped at 160 characters; user-supplied text can influence only that user's own data.
- Dates use `TRIPWIRE_TZ` (default: server local time).

## License
MIT. See `LICENSE`.
