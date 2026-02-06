from __future__ import annotations

from collections.abc import Callable

from .models import ToolAction, ToolKind, ToolResult
from .world_model import MedicalWorldModel


ToolHandler = Callable[[ToolAction], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[ToolKind, ToolHandler] = {}

    def register(self, kind: ToolKind, handler: ToolHandler) -> None:
        self._handlers[kind] = handler

    def invoke(self, action: ToolAction) -> ToolResult:
        if action.kind not in self._handlers:
            raise ValueError(f"No handler registered for tool: {action.kind}")
        return self._handlers[action.kind](action)


def build_default_registry(world_model: MedicalWorldModel) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolKind.ASK_QUESTION, world_model.step)
    registry.register(ToolKind.ORDER_TEST, world_model.step)
    registry.register(ToolKind.RECOMMEND_PLAN, world_model.step)
    return registry
