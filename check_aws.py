"""Verify your AWS setup end-to-end (needs real credentials, a Bedrock model you have access to,
and the DynamoDB table from infra/template.yaml).

    export AWS_REGION=us-east-1 BEDROCK_MODEL_ID=<model-or-inference-profile-id>
    export TRIPWIRE_STORE=dynamodb TRIPWIRE_AI=bedrock
    python scripts/check_aws.py

Uses a throw-away household that is deleted afterwards, so it is safe to re-run.
"""

import os
import sys
import time
import uuid

os.environ.setdefault("TRIPWIRE_AI", "bedrock")  # never silently fall back in this check

from tripwire.service import Tripwire  # noqa: E402
from tripwire.store import make_store  # noqa: E402

store = make_store()
household = f"aws-check-{uuid.uuid4().hex[:8]}"
try:
    store.ping()
    print("DynamoDB reachable.")
    t0 = time.time()
    rec = Tripwire(store, household).record_decision(
        "We're going with the 200 megabit internet plan because it's cheapest. "
        "We'd switch if the price goes above $50 or if outages hit more than twice a month.")
    print(f"[{time.time() - t0:.1f}s]", rec["spoken"])
    if not rec["ok"] or rec["engine"] != "bedrock" or not any(t.get("metric") for t in rec["decision"]["tripwires"]):
        sys.exit(f"FAILED at record_decision (need engine=bedrock and at least one numeric tripwire): {rec}")
    t0 = time.time()
    fact = Tripwire(make_store(), household).report_fact("The provider raised our internet price to $55")  # fresh object => reads DynamoDB
    print(f"[{time.time() - t0:.1f}s]", fact["spoken"])
    if fact.get("engine") != "bedrock" or fact["tripped"] != [rec["decision"]["title"]]:
        sys.exit(f"FAILED at report_fact: {fact}")
    print("OK: Bedrock extraction + assessment and DynamoDB persistence (incl. conditional writes) all work.")
finally:
    print(f"cleaned up {store.purge(household)} item(s).")
