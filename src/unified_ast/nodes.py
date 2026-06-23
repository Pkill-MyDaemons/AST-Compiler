"""Top-level declaration nodes for the unified AST."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .types import UnifiedType, T_INFERRED, T_VOID
from .expr import Block, stmt_from_dict


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"


class TypeDefCategory(str, Enum):
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"


@dataclass
class Param:
    name: str
    type: UnifiedType = field(default_factory=lambda: T_INFERRED)
    default: Optional[str] = None
    is_self: bool = False
    is_variadic: bool = False
    is_keyword_only: bool = False

    def to_dict(self) -> dict:
        d: dict = {"name": self.name}
        if self.type.kind.value != "inferred":
            d["type"] = self.type.to_dict()
        if self.default is not None:
            d["default"] = self.default
        if self.is_self:
            d["is_self"] = True
        if self.is_variadic:
            d["is_variadic"] = True
        if self.is_keyword_only:
            d["is_keyword_only"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Param:
        return cls(
            name=d["name"],
            type=UnifiedType.from_dict(d["type"]) if "type" in d else T_INFERRED,
            default=d.get("default"),
            is_self=d.get("is_self", False),
            is_variadic=d.get("is_variadic", False),
            is_keyword_only=d.get("is_keyword_only", False),
        )


@dataclass
class ImportNode:
    id: str
    module: str
    items: Optional[List[str]] = None
    alias: Optional[str] = None

    @property
    def kind(self) -> str:
        return "import"

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "kind": self.kind, "module": self.module}
        if self.items:
            d["items"] = self.items
        if self.alias is not None:
            d["alias"] = self.alias
        return d


@dataclass
class VariableNode:
    id: str
    name: str
    visibility: Visibility = Visibility.PUBLIC
    is_const: bool = False
    is_static: bool = False
    type: UnifiedType = field(default_factory=lambda: T_INFERRED)
    value: Optional[str] = None

    @property
    def kind(self) -> str:
        return "variable"

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "kind": self.kind, "name": self.name}
        if self.visibility != Visibility.PUBLIC:
            d["visibility"] = self.visibility.value
        if self.is_const:
            d["is_const"] = True
        if self.is_static:
            d["is_static"] = True
        if self.type.kind.value != "inferred":
            d["type"] = self.type.to_dict()
        if self.value is not None:
            d["value"] = self.value
        return d


@dataclass
class FieldNode:
    id: str
    name: str
    type: UnifiedType = field(default_factory=lambda: T_INFERRED)
    visibility: Visibility = Visibility.PUBLIC
    default: Optional[str] = None
    is_mutable: bool = True

    @property
    def kind(self) -> str:
        return "field"

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "kind": self.kind, "name": self.name}
        if self.type.kind.value != "inferred":
            d["type"] = self.type.to_dict()
        if self.visibility != Visibility.PUBLIC:
            d["visibility"] = self.visibility.value
        if self.default is not None:
            d["default"] = self.default
        if not self.is_mutable:
            d["is_mutable"] = False
        return d


@dataclass
class FunctionNode:
    id: str
    name: str
    params: List[Param] = field(default_factory=list)
    return_type: UnifiedType = field(default_factory=lambda: T_INFERRED)
    body: Block = field(default_factory=Block)
    visibility: Visibility = Visibility.PUBLIC
    is_async: bool = False
    is_static: bool = False
    is_constructor: bool = False
    is_abstract: bool = False
    decorators: List[str] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "function"

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "body": self.body.to_dict(),
        }
        if self.visibility != Visibility.PUBLIC:
            d["visibility"] = self.visibility.value
        if self.params:
            d["params"] = [p.to_dict() for p in self.params]
        if self.return_type.kind.value != "inferred":
            d["return_type"] = self.return_type.to_dict()
        if self.is_async:
            d["is_async"] = True
        if self.is_static:
            d["is_static"] = True
        if self.is_constructor:
            d["is_constructor"] = True
        if self.is_abstract:
            d["is_abstract"] = True
        if self.decorators:
            d["decorators"] = self.decorators
        if self.type_params:
            d["type_params"] = self.type_params
        if self.docstring:
            d["docstring"] = self.docstring
        if self.attributes:
            d["attributes"] = self.attributes
        return d


@dataclass
class TypeDefNode:
    id: str
    name: str
    category: TypeDefCategory = TypeDefCategory.CLASS
    visibility: Visibility = Visibility.PUBLIC
    bases: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)
    fields: List[FieldNode] = field(default_factory=list)
    methods: List[FunctionNode] = field(default_factory=list)
    inner_types: List[TypeDefNode] = field(default_factory=list)
    docstring: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "type_def"

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "kind": self.kind, "name": self.name}
        if self.category != TypeDefCategory.CLASS:
            d["category"] = self.category.value
        if self.visibility != Visibility.PUBLIC:
            d["visibility"] = self.visibility.value
        if self.bases:
            d["bases"] = self.bases
        if self.interfaces:
            d["interfaces"] = self.interfaces
        if self.type_params:
            d["type_params"] = self.type_params
        if self.fields:
            d["fields"] = [f.to_dict() for f in self.fields]
        if self.methods:
            d["methods"] = [m.to_dict() for m in self.methods]
        if self.inner_types:
            d["inner_types"] = [t.to_dict() for t in self.inner_types]
        if self.docstring:
            d["docstring"] = self.docstring
        if self.attributes:
            d["attributes"] = self.attributes
        return d


ASTNode = Union[ImportNode, VariableNode, FunctionNode, TypeDefNode]


@dataclass
class Module:
    version: str = "1.0"
    source_language: str = ""
    source_file: str = ""
    nodes: List[ASTNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "source_language": self.source_language,
            "source_file": self.source_file,
            "nodes": [n.to_dict() for n in self.nodes],
        }
        if self.version != "1.0":
            d["version"] = self.version
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Module:
        return cls(
            version=d.get("version", "1.0"),
            source_language=d.get("source_language", ""),
            source_file=d.get("source_file", ""),
            nodes=[_node_from_dict(n) for n in d.get("nodes", [])],
        )


def _node_from_dict(d: dict) -> ASTNode:
    k = d["kind"]
    if k == "import":
        return ImportNode(id=d["id"], module=d["module"], items=d.get("items"), alias=d.get("alias"))
    if k == "variable":
        return VariableNode(
            id=d["id"],
            name=d["name"],
            visibility=Visibility(d.get("visibility", "public")),
            is_const=d.get("is_const", False),
            is_static=d.get("is_static", False),
            type=UnifiedType.from_dict(d["type"]) if "type" in d else T_INFERRED,
            value=d.get("value"),
        )
    if k == "function":
        body_d = d.get("body", {"kind": "block", "stmts": []})
        body = stmt_from_dict(body_d)
        if not isinstance(body, Block):
            from .expr import Block as B
            body = B(stmts=[body])
        return FunctionNode(
            id=d["id"],
            name=d["name"],
            params=[Param.from_dict(p) for p in d.get("params", [])],
            return_type=UnifiedType.from_dict(d["return_type"]) if "return_type" in d else T_INFERRED,
            body=body,
            visibility=Visibility(d.get("visibility", "public")),
            is_async=d.get("is_async", False),
            is_static=d.get("is_static", False),
            is_constructor=d.get("is_constructor", False),
            is_abstract=d.get("is_abstract", False),
            decorators=d.get("decorators", []),
            type_params=d.get("type_params", []),
            docstring=d.get("docstring"),
            attributes=d.get("attributes", {}),
        )
    if k == "type_def":
        return TypeDefNode(
            id=d["id"],
            name=d["name"],
            category=TypeDefCategory(d.get("category", "class")),
            visibility=Visibility(d.get("visibility", "public")),
            bases=d.get("bases", []),
            interfaces=d.get("interfaces", []),
            type_params=d.get("type_params", []),
            fields=[_field_from_dict(f) for f in d.get("fields", [])],
            methods=[_node_from_dict(m) for m in d.get("methods", [])],
            inner_types=[_node_from_dict(t) for t in d.get("inner_types", [])],
            docstring=d.get("docstring"),
            attributes=d.get("attributes", {}),
        )
    raise ValueError(f"Unknown node kind: {k!r}")


def _field_from_dict(d: dict) -> FieldNode:
    return FieldNode(
        id=d["id"],
        name=d["name"],
        type=UnifiedType.from_dict(d["type"]) if "type" in d else T_INFERRED,
        visibility=Visibility(d.get("visibility", "public")),
        default=d.get("default"),
        is_mutable=d.get("is_mutable", True),
    )
