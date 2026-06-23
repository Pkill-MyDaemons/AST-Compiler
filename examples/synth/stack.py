"""Stack data structure."""
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class Stack:
    """A LIFO stack."""

    items: List[int]
    capacity: int

    def __init__(self, capacity: int) -> None:
        self.items = []
        self.capacity = capacity

    def push(self, value: int) -> bool:
        if len(self.items) >= self.capacity:
            return False
        self.items.append(value)
        return True

    def pop(self) -> Optional[int]:
        if not self.items:
            return None
        return self.items.pop()

    def peek(self) -> Optional[int]:
        if not self.items:
            return None
        return self.items[-1]

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def size(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items = []


def transfer(source: Stack, destination: Stack) -> int:
    moved: int = 0
    while not source.is_empty():
        value = source.pop()
        if value is not None:
            destination.push(value)
            moved = moved + 1
    return moved


def build_stack(values: List[int], capacity: int) -> Stack:
    stack = Stack(capacity)
    for v in values:
        stack.push(v)
    return stack
