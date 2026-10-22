# Architecture & design decisions

```
Alexa+ ──(MCP / Streamable HTTP, stateless)──▶ Tripwire server (Starlette + MCP SDK)
                                                  │  service.py  ← all state changes + numeric comparisons
                                                  ├─ ai.py       ← Bedrock Converse, forced tool use (+ offline fallback)
                                                  └─ store.py    ← DynamoDB (prod) / SQLite (local)
```

## 1. The model proposes; the server disposes
`ai.py` asks Bedrock to (a) extract a structured decision and (b) relate a new fact to armed tripwires and report the
observed number. `service.py` then:
- re-validates every field (`validate_extraction`, `validate_assessment`), ignoring unknown tripwire ids;
- performs the numeric comparison itself (`evaluate`), **ignoring the model's verdict** for numeric tripwires;
- never auto-fires qualitative tripwires: they become `pending` and require `confirm_tripwire` from the human.
Tests `test_a_lying_model_cannot_trip_a_numeric_tripwire` and `test_a_model_cannot_untrip_a_real_breach_or_invent_ids`
pin this down (and fail if you make the server trust the model).

## 2. Forced tool use for structured output
Both Bedrock calls set `toolChoice: {tool: {name: …}}` with a JSON schema and `temperature: 0`, so the model must answer
through a schema-shaped function call rather than free text.

## 3. Stateless transport, external state
The MCP endpoint is stateless (no session affinity), so it scales horizontally; *all* memory is in DynamoDB. A tripwire set in
one conversation fires in another because nothing lives in process memory.

## 4. Voice-first contracts
Every tool returns a short `spoken` field safe to read aloud; errors are sentences, never stack traces. Topics are resolved
from natural speech ("the internet thing") with token overlap; **ties produce a clarifying question** instead of a guess. A
limit change with several tripped limits asks *which one*.

## 5. DynamoDB single-table design
`pk = HOUSEHOLD#<id>`, `sk = decision#<slug>`; one small JSON document per decision containing tripwires and history. Access
patterns (list all, get one) need only `Query` and `GetItem`.

## 6. Fallback transparency
`TRIPWIRE_AI=auto` falls back to an offline parser when Bedrock is unavailable and **reports** `engine: offline-fallback`.
`bedrock` mode never falls back, so nothing is silently faked in a production check (`scripts/check_aws.py`).

## 7. Concurrency model
The server is stateless and may run as several containers, so two requests can edit the same decision at once. Each document has a
version number; `Store.update(fn)` reads (doc, version), applies `fn` to a private copy and writes with `expected_ver`; on a
`ConflictError` it re-reads and re-applies. `fn` is therefore a pure function of its input: `_apply_observations` skips tripwires that
are no longer armed, so re-applying (a retry, or a concurrent identical report) is a no-op. DynamoDB uses `UpdateItem` with a
`ConditionExpression`; SQLite uses an in-process lock. `tests/test_robustness.py` forces the worst case with a store that makes two
writers read the same version before either writes, and a mutation check confirms the tests fail if the compare-and-swap is removed.

## 8. Why tripwires have stable ids
An assessment is computed *before* the write (a Bedrock call can take seconds). If tripwires were addressed by list position, an edit
in between would shift positions and a verdict could land on the wrong tripwire (or raise). Ids are generated at creation and the
apply step looks tripwires up by id, ignoring any that no longer exist.

## 9. Failure policy
* Timeouts are explicit (botocore's defaults can hang a request for about a minute and retry five times).
* A circuit breaker bounds the cost of a Bedrock outage; `auto` degrades to the offline parser, `bedrock` fails fast.
* Every service call goes through `server.call()`, which turns any exception into a short spoken sentence and logs the trace.
* `/readyz` actually exercises the database; `/healthz` does not (so a DB blip doesn't restart healthy containers).
* Test strategy: unit tests, moto-backed DynamoDB tests, Stubber-validated Bedrock requests, adversarial model output, forced races,
  fuzzed input, a static IAM check, and a meta-test that fails if a test function is accidentally defined twice (silent shadowing hid a
  weak test during development).
