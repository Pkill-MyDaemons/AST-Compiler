from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class TypeKind(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    BYTES = "bytes"
    LIST = "list"
    MAP = "map"
    SET = "set"
    OPTIONAL = "optional"
    TUPLE = "tuple"
    VOID = "void"
    ANY = "any"
    SELF = "self"
    INFERRED = "inferred"
    NAMED = "named"
    FUNCTION = "function"
    GENERIC = "generic"


@dataclass
class UnifiedType:
    kind: TypeKind
    # NUMBER metadata
    bits: Optional[int] = None
    signed: Optional[bool] = None
    float: Optional[bool] = None
    # NAMED
    name: Optional[str] = None
    # LIST / SET / OPTIONAL — single element type
    element: Optional[UnifiedType] = None
    # MAP
    key: Optional[UnifiedType] = None
    value: Optional[UnifiedType] = None
    # TUPLE
    elements: Optional[List[UnifiedType]] = None
    # FUNCTION
    params: Optional[List[UnifiedType]] = None
    ret: Optional[UnifiedType] = None
    # GENERIC type parameter name (e.g. "T")
    type_params: Optional[List[str]] = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind.value}
        if self.bits is not None:
            d["bits"] = self.bits
        if self.signed is not None:
            d["signed"] = self.signed
        if self.float is not None:
            d["float"] = self.float
        if self.name is not None:
            d["name"] = self.name
        if self.element is not None:
            d["element"] = self.element.to_dict()
        if self.key is not None:
            d["key"] = self.key.to_dict()
        if self.value is not None:
            d["value"] = self.value.to_dict()
        if self.elements is not None:
            d["elements"] = [e.to_dict() for e in self.elements]
        if self.params is not None:
            d["params"] = [p.to_dict() for p in self.params]
        if self.ret is not None:
            d["ret"] = self.ret.to_dict()
        if self.type_params is not None:
            d["type_params"] = self.type_params
        return d

    @classmethod
    def from_dict(cls, d: dict) -> UnifiedType:
        kind = TypeKind(d["kind"])
        return cls(
            kind=kind,
            bits=d.get("bits"),
            signed=d.get("signed"),
            float=d.get("float"),
            name=d.get("name"),
            element=cls.from_dict(d["element"]) if "element" in d else None,
            key=cls.from_dict(d["key"]) if "key" in d else None,
            value=cls.from_dict(d["value"]) if "value" in d else None,
            elements=[cls.from_dict(e) for e in d["elements"]] if "elements" in d else None,
            params=[cls.from_dict(p) for p in d["params"]] if "params" in d else None,
            ret=cls.from_dict(d["ret"]) if "ret" in d else None,
            type_params=d.get("type_params"),
        )

    def render(self) -> str:
        """Human-readable type string for skeleton views."""
        k = self.kind
        if k == TypeKind.NUMBER:
            if self.bits:
                prefix = "f" if self.float else ("i" if self.signed else "u")
                return f"number({prefix}{self.bits})"
            return "number"
        if k == TypeKind.STRING:
            return "string"
        if k == TypeKind.BOOLEAN:
            return "boolean"
        if k == TypeKind.BYTES:
            return "bytes"
        if k == TypeKind.LIST:
            inner = self.element.render() if self.element else "?"
            return f"list<{inner}>"
        if k == TypeKind.MAP:
            kk = self.key.render() if self.key else "?"
            vv = self.value.render() if self.value else "?"
            return f"map<{kk},{vv}>"
        if k == TypeKind.SET:
            inner = self.element.render() if self.element else "?"
            return f"set<{inner}>"
        if k == TypeKind.OPTIONAL:
            inner = self.element.render() if self.element else "?"
            return f"optional<{inner}>"
        if k == TypeKind.TUPLE:
            parts = [e.render() for e in self.elements] if self.elements else []
            return f"tuple<{','.join(parts)}>"
        if k == TypeKind.VOID:
            return "void"
        if k == TypeKind.ANY:
            return "any"
        if k == TypeKind.SELF:
            return "self"
        if k == TypeKind.INFERRED:
            return "_"
        if k == TypeKind.NAMED:
            return self.name or "?"
        if k == TypeKind.FUNCTION:
            ps = [p.render() for p in self.params] if self.params else []
            r = self.ret.render() if self.ret else "void"
            return f"fn({','.join(ps)})->{r}"
        if k == TypeKind.GENERIC:
            return self.name or "T"
        return k.value


# Convenience constructors
def T_NUMBER(bits: Optional[int] = None, signed: bool = True, float: bool = False) -> UnifiedType:
    return UnifiedType(TypeKind.NUMBER, bits=bits, signed=signed, float=float)

T_STRING = UnifiedType(TypeKind.STRING)
T_BOOLEAN = UnifiedType(TypeKind.BOOLEAN)
T_BYTES = UnifiedType(TypeKind.BYTES)
T_VOID = UnifiedType(TypeKind.VOID)
T_ANY = UnifiedType(TypeKind.ANY)
T_SELF = UnifiedType(TypeKind.SELF)
T_INFERRED = UnifiedType(TypeKind.INFERRED)


def T_LIST(elem: UnifiedType) -> UnifiedType:
    return UnifiedType(TypeKind.LIST, element=elem)


def T_MAP(k: UnifiedType, v: UnifiedType) -> UnifiedType:
    return UnifiedType(TypeKind.MAP, key=k, value=v)


def T_SET(elem: UnifiedType) -> UnifiedType:
    return UnifiedType(TypeKind.SET, element=elem)


def T_OPTIONAL(inner: UnifiedType) -> UnifiedType:
    return UnifiedType(TypeKind.OPTIONAL, element=inner)


def T_TUPLE(*elems: UnifiedType) -> UnifiedType:
    return UnifiedType(TypeKind.TUPLE, elements=list(elems))


def T_NAMED(name: str) -> UnifiedType:
    return UnifiedType(TypeKind.NAMED, name=name)


def T_GENERIC(name: str) -> UnifiedType:
    return UnifiedType(TypeKind.GENERIC, name=name)
