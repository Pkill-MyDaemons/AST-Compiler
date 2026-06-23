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
