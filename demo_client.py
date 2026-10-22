"""Narrated terminal demo: drives a running server through the conversation shown in the video.

    TRIPWIRE_AI=offline tripwire &       # (no AWS needed)  -- or configure Bedrock, see README
    python scripts/demo_client.py [http://127.0.0.1:8000/mcp]
"""

import asyncio
import os
import sys

from mcp import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/mcp"
HEADERS = {"Authorization": f"Bearer {os.environ['TRIPWIRE_TOKEN']}"} if os.environ.get("TRIPWIRE_TOKEN") else None

SCRIPT = [
    ("Alexa, remember this: we're going with the 200 megabit internet plan because it's cheapest for our usage. "
     "We'd switch if the price goes above $50 or if outages hit more than twice a month. Check back in three months.",
     "record_decision", {"what_was_said": "We're going with the 200 megabit internet plan because it's cheapest for our usage. "
                         "We'd switch if the price goes above $50 or if outages hit more than twice a month. Check back in three months."}),
    ("Alexa, we're getting the grey sofa because it's comfy. We'd return it if the price drops below $300.",
     "record_decision", {"what_was_said": "We're going with the grey sofa because it's comfy. We'd return it if the price drops below $300."}),
    ("(two weeks later) Alexa, the provider emailed: our internet price goes to $48 next month.",
     "report_fact", {"statement": "The provider emailed that our internet price goes to $48 next month"}),
    ("(a week later) Alexa, the provider just raised our internet price to $55.",
     "report_fact", {"statement": "The provider just raised our internet price to $55"}),
    ("Alexa, what should I look at?", "whats_on_watch", {}),
    ("Alexa, why did we pick that internet plan?", "why_did_we", {"topic": "internet plan"}),
    ("Alexa, keep it, but move the price limit to $60.", "resolve_decision",
     {"topic": "internet", "outcome": "keep", "note": "still best value", "new_limit": 60, "tripwire_number": 1}),
    ("Alexa, the sofa is on sale, the price dropped to $250.", "report_fact", {"statement": "The sofa price dropped to $250"}),
]


async def main() -> int:
    async with (Client(URL, headers=HEADERS) if HEADERS else Client(URL)) as c:
        existing = (await c.call_tool("whats_on_watch", {})).structured_content.get("decisions", [])
        if existing:
            print(f"The server already holds {len(existing)} decision(s); the demo needs a clean slate.\n"
                  "Restart it with a fresh TRIPWIRE_DB (delete the file first), e.g.  rm /tmp/demo.db; TRIPWIRE_DB=/tmp/demo.db tripwire")
            return 1
        for said, tool, args in SCRIPT:
            r = await c.call_tool(tool, args)
            print(f"\n🗣  {said}\n   [{tool}]\n🔊 {r.structured_content['spoken']}")
    return 0


sys.exit(asyncio.run(main()))
