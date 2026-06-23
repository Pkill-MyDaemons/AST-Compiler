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
