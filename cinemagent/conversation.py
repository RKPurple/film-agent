"""
cinemagent.conversation -- multi-turn conversations over run_agent().

Gemini holds the model context server-side (store=True), so a conversation
is just a pointer: each turn's first create() chains from the last
COMPLETED turn's interaction id via previous_interaction_id.

- last_interaction_id only advances on a resumable turn (done.resumable,
  i.e. exit_reason final_answer or max_iterations_answered). After an
  error, an unanswered iteration cap or an abort, the last stored
  interaction may be waiting on tool results that were never sent, so the
  next question chains from the last completed turn instead.
- turn counts every question asked in this conversation, including failed
  and aborted ones. It's incremented when a turn starts, so every trace
  record gets a distinct turn number even when some turns don't complete;
  whether the NEXT turn continues from a given one is tracked separately by
  last_interaction_id.
- Conversation doesn't own the ToolContext or client -- the caller opens
  and closes them.
"""

import uuid

from cinemagent.agent import TRACE_LOG_PATH, run_agent, with_trace_log
from cinemagent.config import AGENT_MAX_ITERATIONS


class Conversation:
    def __init__(self, client, ctx, max_iterations=AGENT_MAX_ITERATIONS):
        self.client = client
        self.ctx = ctx
        self.max_iterations = max_iterations
        self.reset()

    def reset(self):
        """Start a fresh conversation: nothing is chained from before."""
        self.conversation_id = str(uuid.uuid4())
        self.turn = 0
        self.last_interaction_id = None

    def ask(self, question, log_path=TRACE_LOG_PATH):
        """Run one turn, yielding its events unchanged (see
        cinemagent.agent). The turn is trace-logged with this conversation's
        metadata. last_interaction_id is updated before done is yielded, so
        a consumer handling done already sees the new state."""
        self.turn += 1
        previous = self.last_interaction_id
        events = with_trace_log(
            question,
            run_agent(question, self.client, self.ctx, max_iterations=self.max_iterations,
                      previous_interaction_id=previous),
            log_path=log_path,
            conversation_id=self.conversation_id,
            turn=self.turn,
            previous_interaction_id=previous,
        )
        try:
            for event in events:
                if event["type"] == "done" and event["resumable"]:
                    self.last_interaction_id = event["interaction_id"]
                yield event
        finally:
            # Closing ask() early must close the wrapped stream too, so its
            # trace record is written as "aborted" right away.
            events.close()
