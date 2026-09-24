"""
Multi-turn chaining: run_agent(previous_interaction_id=...) and
cinemagent.conversation.Conversation, driven by the scripted fake client
(tests/fakes.py) -- no network.
"""

import json
from contextlib import closing

import pytest
from google.genai import errors as genai_errors

from cinemagent.agent import run_agent
from cinemagent.conversation import Conversation
from fakes import FakeClient, answer_response, tool_response

ELBA = ("filter_by_actor", {"name": "Idris Elba"})
ELBA_RESULT = {"films": [{"film_id": 10195, "title": "Thor", "year": 2011, "my_rating": 3.5}]}


@pytest.fixture(autouse=True)
def elba_tool(fake_tools):
    fake_tools.results["filter_by_actor"] = ELBA_RESULT
    return fake_tools


def first_create_ids(client):
    """previous_interaction_id of every create() call, in order."""
    return [call["previous_interaction_id"] for call in client.create_calls]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ask(conversation, question, trace_path):
    return list(conversation.ask(question, log_path=trace_path))


def server_error():
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})


# --- a. run_agent chains its first call from previous_interaction_id --------

def test_run_agent_chains_from_previous_interaction_id():
    client = FakeClient([tool_response("r1", ELBA), answer_response("r2", "Thor (2011).")])
    events = list(run_agent("q", client, ctx=None, previous_interaction_id="prev_1"))

    assert first_create_ids(client) == ["prev_1", "r1"]
    done = events[-1]
    assert done["exit_reason"] == "final_answer" and done["interaction_id"] == "r2"


def test_run_agent_error_on_first_call_reports_no_interaction_id():
    # done.interaction_id is this turn's own latest response, never the
    # incoming previous_interaction_id.
    client = FakeClient([server_error()])
    done = list(run_agent("q", client, ctx=None, previous_interaction_id="prev_1"))[-1]
    assert done["exit_reason"] == "error" and done["interaction_id"] is None


# --- b. two-turn happy path ---------------------------------------------------

def test_conversation_two_turns(trace_path):
    client = FakeClient([
        tool_response("t1_r1", ELBA), answer_response("t1_r2", "You watched Thor (2011)."),
        answer_response("t2_r1", "Thor is your highest rated."),
    ])
    conv = Conversation(client, ctx=None)

    ask(conv, "Idris Elba movies?", trace_path)
    assert conv.turn == 1 and conv.last_interaction_id == "t1_r2"
    ask(conv, "Which did I rate highest?", trace_path)

    assert first_create_ids(client) == [None, "t1_r1", "t1_r2"]
    assert conv.turn == 2
    assert conv.last_interaction_id == "t2_r1"


# --- c/d. a non-resumable turn 2 doesn't move the chain ---------------------

@pytest.mark.parametrize("turn2_script, exit_reason", [
    ([tool_response("t2_r1", ELBA), server_error()], "error"),
    ([tool_response("t2_r1", ("filter_by_actor", {"name": "A"})),
      tool_response("t2_r2", ("filter_by_actor", {"name": "B"})),
      server_error()], "max_iterations_reached"),                      # the no-tools final call fails
], ids=["api_error", "iteration_cap_unanswered"])
def test_non_resumable_turn_keeps_last_completed_id(trace_path, turn2_script, exit_reason):
    client = FakeClient([answer_response("t1_r1", "Turn 1 answer."), *turn2_script,
                         answer_response("t3_r1", "Turn 3 answer.")])
    conv = Conversation(client, ctx=None, max_iterations=2)

    ask(conv, "turn 1", trace_path)
    done2 = ask(conv, "turn 2", trace_path)[-1]
    assert done2["exit_reason"] == exit_reason and done2["resumable"] is False
    assert conv.last_interaction_id == "t1_r1"
    assert conv.turn == 2

    ask(conv, "turn 3", trace_path)
    turn3_first_call = client.create_calls[-1]
    assert turn3_first_call["previous_interaction_id"] == "t1_r1"
    assert turn3_first_call["input"] == "turn 3"
    assert conv.turn == 3 and conv.last_interaction_id == "t3_r1"


def test_answered_iteration_cap_advances_chain(trace_path):
    client = FakeClient([answer_response("t1_r1", "Turn 1 answer."),
                         tool_response("t2_r1", ("filter_by_actor", {"name": "A"})),
                         tool_response("t2_r2", ("filter_by_actor", {"name": "B"})),
                         answer_response("t2_r3", "Answer from what I found."),
                         answer_response("t3_r1", "Turn 3 answer.")])
    conv = Conversation(client, ctx=None, max_iterations=2)
    ask(conv, "turn 1", trace_path)
    done2 = ask(conv, "turn 2", trace_path)[-1]
    assert done2["exit_reason"] == "max_iterations_answered" and done2["resumable"] is True
    assert conv.last_interaction_id == "t2_r3"
    ask(conv, "turn 3", trace_path)
    assert client.create_calls[-1]["previous_interaction_id"] == "t2_r3"


# --- e. aborted turn ----------------------------------------------------------

def test_aborted_turn_keeps_last_completed_id(trace_path, elba_tool):
    client = FakeClient([answer_response("t1_r1", "Turn 1 answer."),
                         tool_response("t2_r1", ELBA), answer_response("t2_r2", "never reached")])
    conv = Conversation(client, ctx=None)
    ask(conv, "turn 1", trace_path)

    with closing(conv.ask("turn 2", log_path=trace_path)) as events:
        for event in events:
            if event["type"] == "tool_call":
                break

    assert conv.last_interaction_id == "t1_r1"
    assert conv.turn == 2
    assert elba_tool.count() == 0
    assert records(trace_path)[-1]["exit_reason"] == "aborted"


# --- f. reset -----------------------------------------------------------------

def test_reset_starts_unchained(trace_path):
    client = FakeClient([answer_response("t1_r1", "Turn 1 answer."), answer_response("n1_r1", "Fresh answer.")])
    conv = Conversation(client, ctx=None)
    ask(conv, "turn 1", trace_path)
    old_id = conv.conversation_id

    conv.reset()
    assert conv.conversation_id != old_id
    assert conv.turn == 0 and conv.last_interaction_id is None

    ask(conv, "fresh question", trace_path)
    assert client.create_calls[-1]["previous_interaction_id"] is None
    assert conv.turn == 1


# --- g. trace records carry the conversation metadata -----------------------

def test_trace_records_carry_conversation_metadata(trace_path):
    client = FakeClient([
        answer_response("t1_r1", "Turn 1 answer."),
        server_error(),                                   # turn 2 fails
        answer_response("t3_r1", "Turn 3 answer."),
    ])
    conv = Conversation(client, ctx=None)
    first_id = conv.conversation_id
    for question in ("turn 1", "turn 2", "turn 3"):
        ask(conv, question, trace_path)
    conv.reset()
    client.interactions.script.append(answer_response("n1_r1", "New conversation answer."))
    ask(conv, "new turn 1", trace_path)

    got = [(r["query"], r["conversation_id"], r["turn"], r["previous_interaction_id"], r["exit_reason"])
           for r in records(trace_path)]
    assert got == [
        ("turn 1", first_id, 1, None, "final_answer"),
        ("turn 2", first_id, 2, "t1_r1", "error"),
        ("turn 3", first_id, 3, "t1_r1", "final_answer"),
        ("new turn 1", conv.conversation_id, 1, None, "final_answer"),
    ]
    assert conv.conversation_id != first_id
