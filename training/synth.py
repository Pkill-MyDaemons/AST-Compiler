"""
Synthetic source-file generator.

Produces varied Python programs so the dataset covers:
  - top-level functions (not just methods)
  - multiple class hierarchies
  - different type patterns
  - different method counts
"""
from __future__ import annotations
import random
import textwrap
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TEMPLATES: List[str] = []

TEMPLATES.append('''\
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
''')

TEMPLATES.append('''\
"""Bank account management."""
from typing import List, Optional


class Transaction:
    """A single account transaction."""

    amount: float
    description: str
    transaction_type: str

    def __init__(self, amount: float, description: str, transaction_type: str) -> None:
        self.amount = amount
        self.description = description
        self.transaction_type = transaction_type

    def is_debit(self) -> bool:
        return self.transaction_type == "debit"

    def is_credit(self) -> bool:
        return self.transaction_type == "credit"


class BankAccount:
    """A simple bank account."""

    owner: str
    balance: float
    transactions: List[Transaction]
    account_number: str

    def __init__(self, owner: str, account_number: str, initial_balance: float) -> None:
        self.owner = owner
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []

    def deposit(self, amount: float, description: str) -> float:
        if amount <= 0.0:
            return self.balance
        t = Transaction(amount, description, "credit")
        self.transactions.append(t)
        self.balance = self.balance + amount
        return self.balance

    def withdraw(self, amount: float, description: str) -> bool:
        if amount <= 0.0 or amount > self.balance:
            return False
        t = Transaction(amount, description, "debit")
        self.transactions.append(t)
        self.balance = self.balance - amount
        return True

    def get_history(self) -> List[Transaction]:
        return self.transactions

    def total_credits(self) -> float:
        total: float = 0.0
        for t in self.transactions:
            if t.is_credit():
                total = total + t.amount
        return total

    def total_debits(self) -> float:
        total: float = 0.0
        for t in self.transactions:
            if t.is_debit():
                total = total + t.amount
        return total


def transfer_funds(source: BankAccount, target: BankAccount, amount: float) -> bool:
    if not source.withdraw(amount, "transfer out"):
        return False
    target.deposit(amount, "transfer in")
    return True


def find_largest_transaction(account: BankAccount) -> Optional[Transaction]:
    if not account.transactions:
        return None
    best = account.transactions[0]
    for t in account.transactions:
        if t.amount > best.amount:
            best = t
    return best
''')

TEMPLATES.append('''\
"""Simple task manager."""
from typing import List, Optional


class Task:
    """A single to-do task."""

    title: str
    description: str
    priority: int
    completed: bool
    tags: List[str]

    def __init__(self, title: str, description: str, priority: int) -> None:
        self.title = title
        self.description = description
        self.priority = priority
        self.completed = False
        self.tags = []

    def complete(self) -> None:
        self.completed = True

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> bool:
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


class TaskManager:
    """Manages a collection of tasks."""

    tasks: List[Task]

    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def remove_task(self, title: str) -> bool:
        for i in range(len(self.tasks)):
            if self.tasks[i].title == title:
                self.tasks.pop(i)
                return True
        return False

    def find_task(self, title: str) -> Optional[Task]:
        for task in self.tasks:
            if task.title == title:
                return task
        return None

    def pending_tasks(self) -> List[Task]:
        result: List[Task] = []
        for task in self.tasks:
            if not task.completed:
                result.append(task)
        return result

    def completed_tasks(self) -> List[Task]:
        result: List[Task] = []
        for task in self.tasks:
            if task.completed:
                result.append(task)
        return result

    def by_priority(self) -> List[Task]:
        result: List[Task] = list(self.tasks)
        n: int = len(result)
        i: int = 0
        while i < n - 1:
            j: int = i + 1
            while j < n:
                if result[j].priority > result[i].priority:
                    temp = result[i]
                    result[i] = result[j]
                    result[j] = temp
                j = j + 1
            i = i + 1
        return result

    def tasks_with_tag(self, tag: str) -> List[Task]:
        result: List[Task] = []
        for task in self.tasks:
            if task.has_tag(tag):
                result.append(task)
        return result


def completion_rate(manager: TaskManager) -> float:
    total: int = len(manager.tasks)
    if total == 0:
        return 0.0
    done: int = len(manager.completed_tasks())
    return done / total


def highest_priority_task(tasks: List[Task]) -> Optional[Task]:
    if not tasks:
        return None
    best = tasks[0]
    for t in tasks:
        if t.priority > best.priority:
            best = t
    return best
''')

TEMPLATES.append('''\
"""Vector math utilities."""
from typing import List, Optional
import math


class Vector2D:
    """A 2D vector."""

    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def add(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def subtract(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def scale(self, factor: float) -> "Vector2D":
        return Vector2D(self.x * factor, self.y * factor)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def normalize(self) -> Optional["Vector2D"]:
        mag: float = self.magnitude()
        if mag == 0.0:
            return None
        return Vector2D(self.x / mag, self.y / mag)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0


def distance(a: Vector2D, b: Vector2D) -> float:
    dx: float = a.x - b.x
    dy: float = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def centroid(points: List[Vector2D]) -> Optional[Vector2D]:
    if not points:
        return None
    sum_x: float = 0.0
    sum_y: float = 0.0
    for p in points:
        sum_x = sum_x + p.x
        sum_y = sum_y + p.y
    n: float = len(points)
    return Vector2D(sum_x / n, sum_y / n)


def closest_point(target: Vector2D, candidates: List[Vector2D]) -> Optional[Vector2D]:
    if not candidates:
        return None
    best = candidates[0]
    best_dist: float = distance(target, best)
    for c in candidates:
        d: float = distance(target, c)
        if d < best_dist:
            best_dist = d
            best = c
    return best
''')

TEMPLATES.append('''\
"""Simple key-value cache with TTL."""
from typing import Dict, List, Optional


class CacheEntry:
    """A single cache entry."""

    key: str
    value: str
    ttl: int
    hits: int

    def __init__(self, key: str, value: str, ttl: int) -> None:
        self.key = key
        self.value = value
        self.ttl = ttl
        self.hits = 0

    def is_expired(self) -> bool:
        return self.ttl <= 0

    def tick(self) -> None:
        if self.ttl > 0:
            self.ttl = self.ttl - 1

    def access(self) -> str:
        self.hits = self.hits + 1
        return self.value


class Cache:
    """In-memory key-value cache."""

    entries: Dict[str, CacheEntry]
    max_size: int
    default_ttl: int

    def __init__(self, max_size: int, default_ttl: int) -> None:
        self.entries = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if len(self.entries) >= self.max_size and key not in self.entries:
            return False
        actual_ttl: int = ttl if ttl is not None else self.default_ttl
        self.entries[key] = CacheEntry(key, value, actual_ttl)
        return True

    def get(self, key: str) -> Optional[str]:
        entry = self.entries.get(key)
        if entry is None or entry.is_expired():
            return None
        return entry.access()

    def delete(self, key: str) -> bool:
        if key in self.entries:
            del self.entries[key]
            return True
        return False

    def evict_expired(self) -> int:
        expired: List[str] = []
        for key, entry in self.entries.items():
            if entry.is_expired():
                expired.append(key)
        for key in expired:
            del self.entries[key]
        return len(expired)

    def tick(self) -> None:
        for entry in self.entries.values():
            entry.tick()

    def size(self) -> int:
        return len(self.entries)

    def most_accessed(self) -> Optional[CacheEntry]:
        if not self.entries:
            return None
        best = None
        best_hits: int = -1
        for entry in self.entries.values():
            if entry.hits > best_hits:
                best_hits = entry.hits
                best = entry
        return best


def warm_cache(cache: Cache, data: Dict[str, str]) -> int:
    loaded: int = 0
    for key, value in data.items():
        if cache.set(key, value):
            loaded = loaded + 1
    return loaded
''')

TEMPLATES.append('''\
"""Event system."""
from typing import Callable, Dict, List, Optional


class Event:
    """An event with a name and payload."""

    name: str
    payload: str
    source: str

    def __init__(self, name: str, payload: str, source: str) -> None:
        self.name = name
        self.payload = payload
        self.source = source

    def is_from(self, source: str) -> bool:
        return self.source == source


class EventBus:
    """Publish-subscribe event bus."""

    handlers: Dict[str, List[str]]
    event_log: List[Event]

    def __init__(self) -> None:
        self.handlers = {}
        self.event_log = []

    def subscribe(self, event_name: str, handler_id: str) -> None:
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        if handler_id not in self.handlers[event_name]:
            self.handlers[event_name].append(handler_id)

    def unsubscribe(self, event_name: str, handler_id: str) -> bool:
        if event_name not in self.handlers:
            return False
        if handler_id in self.handlers[event_name]:
            self.handlers[event_name].remove(handler_id)
            return True
        return False

    def publish(self, event: Event) -> int:
        self.event_log.append(event)
        handlers = self.handlers.get(event.name, [])
        return len(handlers)

    def subscriber_count(self, event_name: str) -> int:
        return len(self.handlers.get(event_name, []))

    def history(self, event_name: str) -> List[Event]:
        result: List[Event] = []
        for e in self.event_log:
            if e.name == event_name:
                result.append(e)
        return result

    def clear_log(self) -> int:
        count: int = len(self.event_log)
        self.event_log = []
        return count


def replay_events(bus: EventBus, events: List[Event]) -> int:
    replayed: int = 0
    for event in events:
        bus.publish(event)
        replayed = replayed + 1
    return replayed


def count_events_from(events: List[Event], source: str) -> int:
    count: int = 0
    for e in events:
        if e.is_from(source):
            count = count + 1
    return count
''')

TEMPLATES.append('''\
"""Simple graph with BFS/DFS."""
from typing import Dict, List, Optional, Set


class Graph:
    """Undirected adjacency-list graph."""

    adjacency: Dict[str, List[str]]

    def __init__(self) -> None:
        self.adjacency = {}

    def add_node(self, node: str) -> None:
        if node not in self.adjacency:
            self.adjacency[node] = []

    def add_edge(self, a: str, b: str) -> None:
        self.add_node(a)
        self.add_node(b)
        if b not in self.adjacency[a]:
            self.adjacency[a].append(b)
        if a not in self.adjacency[b]:
            self.adjacency[b].append(a)

    def remove_edge(self, a: str, b: str) -> bool:
        changed: bool = False
        if a in self.adjacency and b in self.adjacency[a]:
            self.adjacency[a].remove(b)
            changed = True
        if b in self.adjacency and a in self.adjacency[b]:
            self.adjacency[b].remove(a)
            changed = True
        return changed

    def neighbors(self, node: str) -> List[str]:
        return self.adjacency.get(node, [])

    def node_count(self) -> int:
        return len(self.adjacency)

    def has_edge(self, a: str, b: str) -> bool:
        return b in self.adjacency.get(a, [])


def bfs(graph: Graph, start: str) -> List[str]:
    if start not in graph.adjacency:
        return []
    visited: List[str] = []
    queue: List[str] = [start]
    seen: List[str] = [start]
    while queue:
        node = queue.pop(0)
        visited.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in seen:
                seen.append(neighbor)
                queue.append(neighbor)
    return visited


def dfs(graph: Graph, start: str) -> List[str]:
    if start not in graph.adjacency:
        return []
    visited: List[str] = []
    stack: List[str] = [start]
    seen: List[str] = []
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.append(node)
            visited.append(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in seen:
                    stack.append(neighbor)
    return visited


def is_connected(graph: Graph) -> bool:
    if graph.node_count() == 0:
        return True
    start = list(graph.adjacency.keys())[0]
    reached = bfs(graph, start)
    return len(reached) == graph.node_count()


def shortest_path_length(graph: Graph, start: str, end: str) -> int:
    if start == end:
        return 0
    queue: List[str] = [start]
    distances: Dict[str, int] = {start: 0}
    while queue:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                if neighbor == end:
                    return distances[neighbor]
                queue.append(neighbor)
    return -1
''')


def write_all(output_dir: str) -> List[str]:
    """Write all synthetic files and return their paths."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    names = ["stack", "bank", "tasks", "vectors", "cache", "events", "graph"]
    paths = []
    for name, template in zip(names, TEMPLATES):
        path = os.path.join(output_dir, f"{name}.py")
        with open(path, "w") as f:
            f.write(template)
        paths.append(path)
    return paths
