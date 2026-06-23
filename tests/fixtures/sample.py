"""Sample Python file for round-trip testing."""
import os
from typing import List, Optional, Dict

MAX_RETRIES: int = 3
DEFAULT_NAME: str = "world"


class Animal:
    """Base class for animals."""

    name: str
    sound: str = "..."

    def __init__(self, name: str, sound: str) -> None:
        self.name = name
        self.sound = sound

    def speak(self) -> str:
        return self.sound

    def describe(self) -> str:
        return f"I am {self.name}"


class Dog(Animal):
    """A dog that can fetch."""

    tricks: List[str]

    def __init__(self, name: str) -> None:
        self.name = name
        self.sound = "woof"
        self.tricks = []

    def learn_trick(self, trick: str) -> None:
        self.tricks.append(trick)

    def perform(self, trick: str) -> bool:
        if trick in self.tricks:
            return True
        return False


def greet(name: str) -> str:
    return "Hello, " + name


def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a: int = 0
    b: int = 1
    i: int = 2
    while i <= n:
        temp: int = a + b
        a = b
        b = temp
        i = i + 1
    return b


def find_max(numbers: List[int]) -> Optional[int]:
    if not numbers:
        return None
    best: int = numbers[0]
    for x in numbers:
        if x > best:
            best = x
    return best


def count_words(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for word in text.split():
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1
    return counts
