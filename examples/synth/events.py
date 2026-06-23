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
