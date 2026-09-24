"""
Deterministic tests of cinemagent.agent's event loop: a scripted fake
Gemini client and fake tools (tests/fakes.py) -- no network, no Gemini, no
Postgres, no models.
"""

import json
import socket
from contextlib import closing

import httpx
import pytest
from google.genai import errors as genai_errors

from google.genai._gaos.lib import compat_errors as interactions_errors

from cinemagent.agent import (DUPLICATE_CALL_ERROR, FALLBACK_MESSAGE, TOOL_BUDGET_NOTE, collect, run_agent,
                              with_trace_log)
from conftest import NetworkBlocked
from fakes import FakeClient, FakeUsage, FakeResponse, answer_response, tool_response

QUERY = "What movies with Idris Elba have I watched?"
ELBA = ("filter_by_actor", {"name": "Idris Elba"})
NOLAN = ("filter_by_director", {"name": "Christopher Nolan"})
ELBA_RESULT = {"films": [{"film_id": 10195, "title": "Thor", "year": 2011, "my_rating": 3.5}]}
NOLAN_RESULT = {"films": [{"film_id": 155, "title": "The Dark Knight", "year": 2008, "my_rating": 5.0}]}

STEP_KEYS = ["tool", "arguments", "result", "blocked_duplicate"]
RECORD_STEP_KEYS = STEP_KEYS + ["id", "iteration", "index", "duration_ms"]
RECORD_FIELDS = ["timestamp", "model", "query", "iterations", "exit_reason", "steps", "answer",
                 "interaction_id", "resumable", "usage", "model_calls", "elapsed_ms", "error", "responses",
                 "conversation_id", "turn", "previous_interaction_id"]


def server_error():
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.",
                                                    "status": "UNAVAILABLE"}})


def client_error(code=400):
    return genai_errors.ClientError(code, {"error": {"code": code, "message": "Bad request.",
                                                     "status": "INVALID_ARGUMENT"}})


def interactions_error(status_code):
    """A real Interactions-client error, as client.interactions.create()
    raises them (NOT google.genai.errors.APIError subclasses)."""
    response = httpx.Response(status_code, request=httpx.Request("POST", "https://example.invalid/interactions"))
    return interactions_errors.APIError.generate(status_code, {"error": {"code": status_code}}, None, response)


def types(events):
    return [e["type"] for e in events]


def run(script, fake_tools, results=None, max_iterations=6):
    """Run one turn against a scripted client; returns (events, client)."""
    fake_tools.results.update(results or {})
    client = FakeClient(script)
    return list(run_agent(QUERY, client, ctx=None, max_iterations=max_iterations)), client


def sent_function_results(create_kwargs):
    """The function_result items the loop sent the model in one create()."""
    return {item["call_id"]: json.loads(item["result"][0]["text"]) for item in create_kwargs["input"]}


# --- scenarios a-g, shared with the serialization test (l) ---------------

def scenario_happy():
    return dict(script=[
        tool_response("r1", ELBA, usage=FakeUsage(total_input_tokens=100, total_output_tokens=10,
                                                  total_thought_tokens=5, total_tokens=115)),
        answer_response("r2", "You watched Thor (2011).",
                        usage=FakeUsage(total_input_tokens=150, total_output_tokens=20, total_thought_tokens=7,
                                        total_cached_tokens=40, total_tokens=177)),
    ], results={"filter_by_actor": ELBA_RESULT})


def scenario_parallel():
    return dict(script=[tool_response("r1", ELBA, NOLAN), answer_response("r2", "Both.")],
                results={"filter_by_actor": ELBA_RESULT, "filter_by_director": NOLAN_RESULT})


def scenario_duplicate_within_response():
    return dict(script=[tool_response("r1", ELBA, ELBA), answer_response("r2", "Thor (2011).")],
                results={"filter_by_actor": ELBA_RESULT})


def scenario_duplicate_across_iterations():
    return dict(script=[tool_response("r1", ELBA), tool_response("r2", ELBA), answer_response("r3", "Thor (2011).")],
                results={"filter_by_actor": ELBA_RESULT})


def cap_script():
    return [tool_response(f"r{i}", ("filter_by_actor", {"name": f"Actor {i}"})) for i in range(1, 4)]


def scenario_cap():
    return dict(script=cap_script() + [answer_response("r4", "From what I found: Actor 1 appears in Thor (2011).")],
                max_iterations=3)


def scenario_cap_fallback():
    return dict(script=cap_script() + [server_error()], max_iterations=3)


def scenario_server_error():
    return dict(script=[tool_response("r1", ELBA), server_error()], results={"filter_by_actor": ELBA_RESULT})


def scenario_client_error():
    return dict(script=[tool_response("r1", ELBA), client_error(400)], results={"filter_by_actor": ELBA_RESULT})


def scenario_timeout():
    return dict(script=[httpx.ReadTimeout("The read operation timed out")])


def scenario_empty_completed():
    return dict(script=[FakeResponse("r1", "completed", output_text=None)])


def scenario_failed_status():
    return dict(script=[FakeResponse("r1", "failed", output_text="partial text")])


ALL_SCENARIOS = [scenario_happy, scenario_parallel, scenario_duplicate_within_response,
                 scenario_duplicate_across_iterations, scenario_cap, scenario_cap_fallback, scenario_server_error,
                 scenario_client_error, scenario_timeout, scenario_empty_completed, scenario_failed_status]


def run_scenario(scenario, fake_tools):
    s = scenario()
    return run(s["script"], fake_tools, s.get("results"), s.get("max_iterations", 6))


# --- a. happy path ----------------------------------------------------------

def test_happy_path(fake_tools):
    events, client = run_scenario(scenario_happy, fake_tools)

    assert types(events) == ["tool_call", "tool_result", "answer", "done"]
    call, result, answer, done = events
    assert call == {"type": "tool_call", "id": "c1", "iteration": 1, "index": 0, "name": "filter_by_actor",
                    "arguments": {"name": "Idris Elba"}, "model_call_id": "r1_fc0"}
    assert result["id"] == "c1" and result["iteration"] == 1 and result["name"] == "filter_by_actor"
    assert result["result"] == ELBA_RESULT and result["blocked_duplicate"] is False
    assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 0
    assert answer == {"type": "answer", "text": "You watched Thor (2011).", "iteration": 2}

    assert done["exit_reason"] == "final_answer"
    assert done["resumable"] is True
    assert done["iterations"] == 2
    assert done["model_calls"] == 2
    assert done["interaction_id"] == "r2"
    assert done["error"] is None
    assert done["usage"] == {"input_tokens": 250, "output_tokens": 30, "thought_tokens": 12,
                             "cached_tokens": 40, "tool_use_tokens": None, "total_tokens": 292}
    assert done["responses"] == [
        {"iteration": 1, "interaction_id": "r1", "status": "requires_action", "function_calls": 1},
        {"iteration": 2, "interaction_id": "r2", "status": "completed", "function_calls": 0},
    ]

    first, second = client.create_calls
    assert first["previous_interaction_id"] is None and first["input"] == QUERY
    assert second["previous_interaction_id"] == "r1"
    assert sent_function_results(second) == {"r1_fc0": ELBA_RESULT}
    assert fake_tools.executions == [("filter_by_actor", {"name": "Idris Elba"})]


# --- b. parallel calls ------------------------------------------------------

def test_parallel_calls(fake_tools):
    events, client = run_scenario(scenario_parallel, fake_tools)

    assert types(events) == ["tool_call", "tool_result", "tool_call", "tool_result", "answer", "done"]
    c1, r1, c2, r2 = events[:4]
    assert (c1["id"], c1["iteration"], c1["index"], c1["name"]) == ("c1", 1, 0, "filter_by_actor")
    assert (c2["id"], c2["iteration"], c2["index"], c2["name"]) == ("c2", 1, 1, "filter_by_director")
    assert r1["id"] == "c1" and r1["result"] == ELBA_RESULT
    assert r2["id"] == "c2" and r2["result"] == NOLAN_RESULT
    assert sent_function_results(client.create_calls[1]) == {"r1_fc0": ELBA_RESULT, "r1_fc1": NOLAN_RESULT}


# --- c. duplicate guard -----------------------------------------------------

def test_duplicate_within_one_response(fake_tools):
    events, client = run_scenario(scenario_duplicate_within_response, fake_tools)

    assert types(events) == ["tool_call", "tool_result", "tool_call", "tool_result", "answer", "done"]
    first, second = events[1], events[3]
    assert first["blocked_duplicate"] is False and first["result"] == ELBA_RESULT
    assert second["blocked_duplicate"] is True and second["result"] == {"error": DUPLICATE_CALL_ERROR}
    assert (events[2]["id"], events[2]["iteration"], events[2]["index"]) == ("c2", 1, 1)
    assert fake_tools.count("filter_by_actor") == 1
    # the model still receives the blocked result
    assert sent_function_results(client.create_calls[1]) == {"r1_fc0": ELBA_RESULT,
                                                             "r1_fc1": {"error": DUPLICATE_CALL_ERROR}}


def test_duplicate_across_iterations(fake_tools):
    events, client = run_scenario(scenario_duplicate_across_iterations, fake_tools)

    assert types(events) == ["tool_call", "tool_result", "tool_call", "tool_result", "answer", "done"]
    assert (events[2]["id"], events[2]["iteration"], events[2]["index"]) == ("c2", 2, 0)
    assert events[1]["blocked_duplicate"] is False
    assert events[3]["blocked_duplicate"] is True and events[3]["result"] == {"error": DUPLICATE_CALL_ERROR}
    assert fake_tools.count("filter_by_actor") == 1
    assert sent_function_results(client.create_calls[2]) == {"r2_fc0": {"error": DUPLICATE_CALL_ERROR}}
    assert events[-1]["exit_reason"] == "final_answer"


# --- d. iteration cap -------------------------------------------------------

def test_iteration_cap_answers_without_tools(fake_tools):
    events, client = run_scenario(scenario_cap, fake_tools)

    assert types(events) == ["tool_call", "tool_result"] * 3 + ["answer", "done"]
    assert [e["id"] for e in events if e["type"] == "tool_call"] == ["c1", "c2", "c3"]
    assert [e["iteration"] for e in events if e["type"] == "tool_call"] == [1, 2, 3]
    answer, done = events[-2:]
    assert answer == {"type": "answer", "text": "From what I found: Actor 1 appears in Thor (2011).", "iteration": 4}
    assert done["exit_reason"] == "max_iterations_answered"
    assert done["resumable"] is True
    assert done["iterations"] == 4 and done["model_calls"] == 4
    assert done["interaction_id"] == "r4"
    assert fake_tools.count() == 3

    # max_iterations + 1 calls; only the last disables function calling
    assert len(client.create_calls) == 4
    assert all("generation_config" not in call for call in client.create_calls[:3])
    final = client.create_calls[-1]
    assert final["generation_config"] == {"tool_choice": "none"}
    assert final["previous_interaction_id"] == "r3"
    # it carries the pending function results from iteration 3, then the budget note
    *results, note = final["input"]
    assert [item["call_id"] for item in results] == ["r3_fc0"]
    assert note == {"type": "user_input", "content": [{"type": "text", "text": TOOL_BUDGET_NOTE}]}
    assert done["responses"][-1] == {"iteration": 4, "interaction_id": "r4", "status": "completed",
                                     "function_calls": 0}


@pytest.mark.parametrize("final_call", [
    pytest.param(lambda: server_error(), id="final_call_raises"),
    pytest.param(lambda: FakeResponse("r4", "completed", output_text=None), id="final_call_empty"),
    pytest.param(lambda: tool_response("r4", ELBA), id="final_call_requests_tools"),
])
def test_iteration_cap_fallback(fake_tools, final_call):
    events, client = run(cap_script() + [final_call()], fake_tools, max_iterations=3)

    assert types(events) == ["tool_call", "tool_result"] * 3 + ["answer", "done"]
    answer, done = events[-2:]
    assert answer == {"type": "answer", "text": FALLBACK_MESSAGE, "iteration": 3}
    assert done["exit_reason"] == "max_iterations_reached"
    assert done["resumable"] is False
    assert done["iterations"] == 3
    assert len(client.create_calls) == 4
    assert client.create_calls[-1]["generation_config"] == {"tool_choice": "none"}
    assert fake_tools.count() == 3  # the final call's tool request (if any) is never executed


# --- e. Gemini API errors ---------------------------------------------------

@pytest.mark.parametrize("scenario, error_type, retryable", [
    (scenario_server_error, "ServerError", True),
    (scenario_client_error, "ClientError", False),
])
def test_api_error_on_second_call(fake_tools, scenario, error_type, retryable):
    events, client = run_scenario(scenario, fake_tools)

    assert types(events) == ["tool_call", "tool_result", "error", "done"]
    error, done = events[2:]
    assert error["where"] == "model_call"
    assert error["iteration"] == 2
    assert error["error_type"] == error_type
    assert error["retryable"] is retryable
    assert error["message"]
    assert done["exit_reason"] == "error" and done["resumable"] is False
    assert done["error"] == error
    assert done["iterations"] == 2 and done["model_calls"] == 1
    assert done["interaction_id"] == "r1"  # last SUCCESSFUL response
    assert len(client.create_calls) == 2

    # collect() keeps the partial trace from iteration 1
    fake_tools.executions.clear()
    answer, trace_steps, done2 = collect(iter(events))
    assert answer is None
    assert trace_steps == [{"tool": "filter_by_actor", "arguments": {"name": "Idris Elba"},
                            "result": ELBA_RESULT, "blocked_duplicate": False}]
    assert done2["exit_reason"] == "error"


def test_rate_limit_client_error_is_retryable(fake_tools):
    events, _ = run([client_error(429)], fake_tools)
    assert types(events) == ["error", "done"]
    assert events[0]["error_type"] == "ClientError" and events[0]["retryable"] is True


@pytest.mark.parametrize("status_code, error_type, retryable", [
    (400, "BadRequestError", False),
    (404, "NotFoundError", False),
    (429, "RateLimitError", True),
    (503, "InternalServerError", True),
])
def test_interactions_client_errors_become_error_events(fake_tools, status_code, error_type, retryable):
    # client.interactions.create() raises these, not google.genai.errors.
    events, _ = run([tool_response("r1", ELBA), interactions_error(status_code)], fake_tools,
                    {"filter_by_actor": ELBA_RESULT})
    assert types(events) == ["tool_call", "tool_result", "error", "done"]
    error = events[2]
    assert (error["where"], error["error_type"], error["retryable"]) == ("model_call", error_type, retryable)
    assert events[-1]["exit_reason"] == "error"


def test_interactions_client_timeout_is_retryable(fake_tools):
    timeout = interactions_errors.APITimeoutError(httpx.Request("POST", "https://example.invalid/interactions"))
    events, _ = run([timeout], fake_tools)
    assert types(events) == ["error", "done"]
    assert (events[0]["error_type"], events[0]["retryable"]) == ("APITimeoutError", True)


# --- f. network error -------------------------------------------------------

def test_network_timeout(fake_tools):
    events, _ = run_scenario(scenario_timeout, fake_tools)

    assert types(events) == ["error", "done"]
    error, done = events
    assert error["where"] == "model_call" and error["iteration"] == 1
    assert error["error_type"] == "ReadTimeout"
    assert error["retryable"] is True
    assert done["exit_reason"] == "error" and done["model_calls"] == 0 and done["interaction_id"] is None


# --- g. unusable responses --------------------------------------------------

@pytest.mark.parametrize("scenario, status", [(scenario_empty_completed, "completed"),
                                              (scenario_failed_status, "failed")])
def test_unusable_response(fake_tools, scenario, status):
    events, _ = run_scenario(scenario, fake_tools)

    assert types(events) == ["error", "done"]
    error, done = events
    assert error["where"] == "model_response"
    assert error["error_type"] == "UnusableResponse"
    assert f"status={status!r}" in error["message"]
    assert done["exit_reason"] == "error" and done["resumable"] is False
    assert done["interaction_id"] == "r1" and done["model_calls"] == 1
    assert done["responses"] == [{"iteration": 1, "interaction_id": "r1", "status": status, "function_calls": 0}]


# --- h. programming errors propagate ----------------------------------------

def test_programming_error_propagates_and_is_logged_as_crashed(fake_tools, trace_path):
    client = FakeClient([tool_response("r1", ELBA), KeyError("boom")])
    fake_tools.results["filter_by_actor"] = ELBA_RESULT
    seen = []
    with pytest.raises(KeyError):
        for event in with_trace_log(QUERY, run_agent(QUERY, client, ctx=None), log_path=trace_path):
            seen.append(event["type"])

    assert seen == ["tool_call", "tool_result"]
    (record,) = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert record["exit_reason"] == "crashed"
    assert [s["tool"] for s in record["steps"]] == ["filter_by_actor"]
    assert record["answer"] is None and record["resumable"] is False


# --- i. abort ---------------------------------------------------------------

def test_abort_after_first_tool_call(fake_tools, trace_path):
    client = FakeClient([tool_response("r1", ELBA), answer_response("r2", "never reached")])
    inner = run_agent(QUERY, client, ctx=None)
    seen = []
    with closing(with_trace_log(QUERY, inner, log_path=trace_path)) as events:
        for event in events:
            seen.append(event["type"])
            if event["type"] == "tool_call":
                break

    assert seen == ["tool_call"]
    assert inner.gi_frame is None  # inner generator closed too
    assert fake_tools.count() == 0  # closed before the tool ran
    assert len(client.create_calls) == 1
    (record,) = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert record["exit_reason"] == "aborted"
    assert record["iterations"] == 1
    assert record["steps"] == [] and record["answer"] is None and record["resumable"] is False


# --- j. trace record shape --------------------------------------------------

def test_trace_record_shape(fake_tools, trace_path):
    s = scenario_parallel()
    fake_tools.results.update(s["results"])
    events = list(with_trace_log(QUERY, run_agent(QUERY, FakeClient(s["script"]), ctx=None), log_path=trace_path))

    (record,) = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert list(record) == RECORD_FIELDS
    assert record["exit_reason"] == "final_answer" and record["resumable"] is True
    assert record["interaction_id"] == "r2" and record["model_calls"] == 2 and record["iterations"] == 2
    assert record["answer"] == "Both." and record["error"] is None
    assert [list(step) for step in record["steps"]] == [RECORD_STEP_KEYS, RECORD_STEP_KEYS]
    assert [(st["id"], st["iteration"], st["index"]) for st in record["steps"]] == [("c1", 1, 0), ("c2", 1, 1)]
    assert record["responses"] == events[-1]["responses"]
    # single-turn use (e.g. the eval): conversation metadata is null
    assert (record["conversation_id"], record["turn"], record["previous_interaction_id"]) == (None, None, None)


# --- k. collect() shape -----------------------------------------------------

def test_collect_trace_steps_have_exactly_four_keys(fake_tools):
    events, _ = run_scenario(scenario_duplicate_within_response, fake_tools)
    answer, trace_steps, done = collect(iter(events))

    assert answer == "Thor (2011)."
    assert [list(step) for step in trace_steps] == [STEP_KEYS, STEP_KEYS]
    assert [step["blocked_duplicate"] for step in trace_steps] == [False, True]
    assert done["exit_reason"] == "final_answer"


# --- l. serialization -------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.__name__.removeprefix("scenario_"))
def test_events_are_plain_json(fake_tools, scenario):
    events, _ = run_scenario(scenario, fake_tools)
    assert events[-1]["type"] == "done"
    for event in events:
        json.dumps(event)  # no default= fallback: must be plain JSON


# --- the network really is blocked ------------------------------------------

def test_network_is_blocked():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("93.184.215.14", 443))
