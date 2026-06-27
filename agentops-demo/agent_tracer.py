import time
import json
import uuid
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field, asdict


# ------------------------------------------------------------------ #
# AgentOps trace structure                                             #
# ------------------------------------------------------------------ #

@dataclass
class ToolCall:
    tool_name: str
    input: Any
    output: Any
    latency_ms: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningStep:
    step_number: int
    thought: str
    action: str  # "tool_call" or "final_answer"
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentTrace:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    final_answer: str = ""
    reasoning_steps: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tool_calls: int = 0
    total_reasoning_steps: int = 0
    success: bool = False
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""

    def add_reasoning_step(self, thought: str, action: str, latency_ms: float):
        step = ReasoningStep(
            step_number=len(self.reasoning_steps) + 1,
            thought=thought,
            action=action,
            latency_ms=latency_ms
        )
        self.reasoning_steps.append(asdict(step))
        self.total_reasoning_steps += 1

    def add_tool_call(self, tool_name: str, input: Any, output: Any,
                      latency_ms: float, success: bool = True):
        call = ToolCall(
            tool_name=tool_name,
            input=input,
            output=output,
            latency_ms=latency_ms,
            success=success
        )
        self.tool_calls.append(asdict(call))
        self.total_tool_calls += 1

    def finish(self, answer: str, success: bool = True):
        self.final_answer = answer
        self.success = success
        self.end_time = datetime.now().isoformat()

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "question": self.question[:80],
            "success": self.success,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_reasoning_steps": self.total_reasoning_steps,
            "total_tool_calls": self.total_tool_calls,
            "tools_used": list(set(c["tool_name"] for c in self.tool_calls)),
            "final_answer": self.final_answer[:100],
        }