"""
Test doubles for cinemagent.agent's loop: a scripted Gemini client and
scripted tools. They mirror only the attributes agent.py actually reads --
interaction.id / .status / .steps (type, name, arguments, id) /
.output_text / .usage.total_* / .errors -- so a change in what agent.py
reads shows up as a test failure, not a silent pass.
"""

USAGE_ATTRS = ("total_input_tokens", "total_output_tokens", "total_thought_tokens",
               "total_cached_tokens", "total_tool_use_tokens", "total_tokens")


class FakeUsage:
    """Like the SDK's Usage: every total_* field present, None if unset."""

    def __init__(self, **totals):
        unknown = set(totals) - set(USAGE_ATTRS)
        if unknown:
            raise TypeError(f"not a real Usage field: {sorted(unknown)}")
        for attr in USAGE_ATTRS:
            setattr(self, attr, totals.get(attr))


class FakeFunctionCall:
    type = "function_call"

    def __init__(self, name, arguments, call_id):
        self.name = name
        self.arguments = arguments
        self.id = call_id


class FakeResponse:
    def __init__(self, response_id, status, steps=(), output_text=None, usage=None, errors=None):
        self.id = response_id
        self.status = status
        self.steps = list(steps)
        self.output_text = output_text
        self.usage = usage
        self.errors = errors


def tool_response(response_id, *calls, usage=None):
    """A response requesting tools. Each call is (name, arguments); Gemini
    call ids are derived from the response id."""
    steps = [FakeFunctionCall(name, args, f"{response_id}_fc{i}") for i, (name, args) in enumerate(calls)]
    return FakeResponse(response_id, "requires_action", steps, usage=usage)


def answer_response(response_id, text, status="completed", usage=None):
    return FakeResponse(response_id, status, output_text=text, usage=usage)


class FakeInteractions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # kwargs of every create() call, in order

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(f"FakeClient script exhausted on create() call #{len(self.calls)}")
        entry = self.script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry


class FakeClient:
    """Stands in for google.genai.Client: interactions.create() returns
    (or raises) the scripted entries one per call."""

    def __init__(self, script):
        self.interactions = FakeInteractions(script)

    @property
    def create_calls(self):
        return self.interactions.calls


class FakeTools:
    """Replaces cinemagent.agent.call_tool. results maps tool name to the
    result dict it returns (or a callable taking the arguments); every
    execution is recorded."""

    def __init__(self, results=None):
        self.results = results or {}
        self.executions = []  # (name, arguments)

    def __call__(self, name, arguments, ctx):
        self.executions.append((name, arguments))
        result = self.results.get(name, {"films": []})
        return result(arguments) if callable(result) else result

    def count(self, name=None):
        return sum(1 for n, _ in self.executions if name is None or n == name)
