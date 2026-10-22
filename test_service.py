from datetime import timedelta

from conftest import INTERNET
from tripwire.service import Tripwire


def test_record_extracts_choice_reason_and_numeric_tripwires(svc):
    r = svc.record_decision(INTERNET)
    d = r["decision"]
    assert r["ok"] and "internet plan" in d["choice"] and d["reason"] == "it's cheapest"
    assert [(t["metric"], t["op"], t["value"]) for t in d["tripwires"]] == [("price", ">", 50.0), ("outages per month", ">", 2.0)]
    assert "1, price above $50" in r["spoken"]


def test_unparseable_decision_gets_a_speakable_hint(svc):
    r = svc.record_decision("hmm what a day")
    assert r["ok"] is False and "we're going with" in r["spoken"].lower()


def test_numeric_tripwire_trips_and_decision_enters_revisit(svc):
    svc.record_decision(INTERNET)
    r = svc.report_fact("The provider raised our internet price to $55")
    assert r["tripped"] == ["internet plan"] and "55" in r["spoken"]
    assert svc.whats_on_watch()["needs_attention"] == ["internet plan"]


def test_non_tripping_value_does_not_change_state_and_warns_when_close(svc):
    svc.record_decision(INTERNET)
    r = svc.report_fact("Our internet price is now $48")
    assert r["tripped"] == [] and r["close_to_limit"] == ["internet plan"] and "close to your limit" in r["spoken"]
    assert "All quiet" in svc.whats_on_watch()["spoken"]
    assert svc.report_fact("internet price is now $30")["close_to_limit"] == []


def test_equality_is_not_above(svc):
    svc.record_decision(INTERNET)
    assert svc.report_fact("the internet price is $50")["tripped"] == []


def test_unrelated_fact_is_ignored(svc):
    svc.record_decision(INTERNET)
    assert svc.report_fact("It rained for 3 days")["tripped"] == []
    assert svc.whats_on_watch()["needs_attention"] == []


def test_fact_routes_to_the_right_decision_when_metrics_collide(svc):
    svc.record_decision(INTERNET)
    svc.record_decision("We're going with the grey sofa because it's comfy. We'd return it if the price goes below $300.")
    r = svc.report_fact("The sofa price dropped to $250")
    assert r["tripped"] == ["grey sofa"]
    assert svc.whats_on_watch()["needs_attention"] == ["grey sofa"]


def test_keep_with_new_limit_rearms_and_never_guesses_which_limit(svc):
    svc.record_decision(INTERNET)
    svc.report_fact("internet price to $55")
    svc.report_fact("we had three outages this month on the internet")
    ask = svc.resolve_decision("internet", "keep", new_limit=60)
    assert ask["ok"] is False and "Which limit" in ask["spoken"]
    ok = svc.resolve_decision("internet", "keep", note="still best value", new_limit=60, tripwire_number=1)
    assert ok["ok"] and "price above $60" in ok["spoken"]
    d = svc.why_did_we("internet")["decision"]
    assert d["status"] == "active" and [t["state"] for t in d["tripwires"]] == ["armed", "armed"]
    assert d["tripwires"][1]["value"] == 2.0  # untouched
    assert svc.report_fact("internet price is now $55")["tripped"] == []  # new limit respected
    assert svc.report_fact("internet price is now $61")["tripped"] == ["internet plan"]


def test_change_and_close_remove_it_from_watch(svc):
    svc.record_decision(INTERNET)
    svc.resolve_decision("internet", "change", note="the 500 megabit plan")
    assert svc.whats_on_watch()["decisions"] == []
    assert svc.report_fact("internet price $99")["tripped"] == []  # no longer watched
    assert "changed" in svc.why_did_we("internet")["spoken"]


def test_revisit_date_surfaces_decision(svc, clock):
    svc.record_decision(INTERNET + " Check back in 3 months.")
    assert "All quiet" in svc.whats_on_watch()["spoken"]
    clock.d += timedelta(days=91)
    assert "check-in date has arrived" in svc.whats_on_watch()["spoken"]


def test_qualitative_tripwire_needs_human_confirmation(svc):
    svc.record_decision("We're going with the blue couch because it's soft.")
    svc.add_tripwire("couch", "the cushions go flat")
    r = svc.report_fact("the couch cushions went flat")
    assert r["needs_confirmation"] == ["blue couch"] and svc.whats_on_watch()["needs_attention"] == []
    assert svc.confirm_tripwire("couch", 1, False)["ok"]
    svc.report_fact("the couch cushions went flat again")
    svc.confirm_tripwire("couch", 1, True)
    assert svc.whats_on_watch()["needs_attention"] == ["blue couch"]


def test_ambiguous_and_unknown_topics_ask_instead_of_guessing(svc):
    svc.record_decision("We're going with the fast internet plan because speed.")
    svc.record_decision("We're going with the cheap phone plan because budget.")
    amb = svc.why_did_we("plan")
    assert amb["ok"] is False and amb.get("ambiguous") and "Which one" in amb["spoken"]
    assert svc.why_did_we("holiday")["ok"] is False


def test_remove_tripwire_and_bad_numbers(svc):
    svc.record_decision(INTERNET)
    assert svc.remove_tripwire("internet", 9)["ok"] is False
    assert svc.remove_tripwire("internet", 2)["ok"]
    assert len(svc.why_did_we("internet")["decision"]["tripwires"]) == 1


def test_why_did_we_tells_the_story(svc):
    svc.record_decision(INTERNET)
    svc.report_fact("internet price to $55")
    s = svc.why_did_we("why did we pick that internet plan")["spoken"]
    for needle in ["October 13", "cheapest", "price above $50", "tripped", "Status: revisit"]:
        assert needle in s


def test_persists_across_instances_and_households_are_isolated(store, clock):
    Tripwire(store, "a", clock).record_decision(INTERNET)
    assert Tripwire(store, "a", clock).report_fact("internet price to $55")["tripped"] == ["internet plan"]
    assert Tripwire(store, "b", clock).whats_on_watch()["decisions"] == []


def test_duplicate_titles_get_unique_keys(svc):
    svc.record_decision(INTERNET)  # title "internet plan"
    svc.record_decision("We're going with the fibre internet plan because it's quick. We'd switch if the price goes above $90.")
    assert sorted(d["_key"] for d in svc.s.list("t", "decision")) == ["internet-plan", "internet-plan-2"]
    assert svc.why_did_we("fibre")["ok"]  # disambiguated by content, even though both are "internet plan"


def test_subjectless_condition_is_dropped_not_turned_into_a_junk_tripwire(svc):
    """'if we go fewer than 4 times a month' has no measurable subject for the offline parser; it must not invent one
    (a junk metric like 'we per month' could fire on unrelated facts)."""
    r = svc.record_decision("We're going with the annual gym membership because it's cheaper. We'd cancel if we go fewer than 4 times a month.")
    assert r["decision"]["tripwires"] == []
    assert svc.report_fact("It rained 3 times a month")["tripped"] == []
