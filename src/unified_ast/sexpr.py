"""S-expression serializer — converts an AST dict to Lisp-style plain text."""
from __future__ import annotations
from typing import Any


def _q(s: Any) -> str:
    """Double-quote a value, escaping special characters."""
    escaped = (
        str(s)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _attr(key: str, val: str) -> str:
    return f"{key}:{_q(val)}"


def _type_str(d: Any) -> str:
    if not isinstance(d, dict):
        return "?"
    kind = d.get("kind", "?")
    if kind == "number":
        bits = d.get("bits", "")
        return f"num{bits}" if bits else "num"
    if kind == "string":
        return "str"
    if kind == "boolean":
        return "bool"
    if kind == "void":
        return "void"
    if kind == "any":
        return "any"
    if kind == "named":
        return str(d.get("name", "?"))
    if kind == "list":
        return f"[{_type_str(d.get('element_type', {}))}]"
    if kind == "optional":
        return f"?{_type_str(d.get('inner', {}))}"
    if kind == "map":
        return f"map[{_type_str(d.get('key_type', {}))}:{_type_str(d.get('value_type', {}))}]"
    if kind == "tuple":
        elems = " ".join(_type_str(e) for e in d.get("elements", []))
        return f"(tup {elems})"
    return kind


def _expr(d: Any) -> str:
    if not isinstance(d, dict):
        return "nil" if d is None else str(d)

    kind = d.get("kind")

    # Identifier compact form: {"name": "x"} with no kind key
    if kind is None and "name" in d:
        return str(d["name"])

    if kind == "identifier":
        return str(d.get("name", "?"))

    if kind == "literal":
        v = d.get("value")
        lk = d.get("lit_kind", "int")
        if lk == "string":
            return _q(v)
        if lk == "bool":
            return "true" if v else "false"
        if v is None:
            return "nil"
        return str(v)

    if kind == "binary_op":
        return f'({d["op"]} {_expr(d["left"])} {_expr(d["right"])})'

    if kind == "unary_op":
        return f'({d["op"]} {_expr(d["operand"])})'

    if kind == "call":
        func_str = _expr(d["func"])
        args = d.get("args", [])
        if args:
            args_str = " ".join(_expr(a) for a in args)
            return f"(call {func_str} [{args_str}])"
        return f"(call {func_str})"

    if kind == "field_access":
        return f'(fa {_expr(d["object"])} {_q(d.get("field", ""))})'

    if kind == "index":
        return f"(idx {_expr(d['object'])} {_expr(d['index'])})"

    if kind == "list_literal":
        elems = " ".join(_expr(e) for e in d.get("elements", []))
        return f"[{elems}]"

    if kind == "dict_literal":
        pairs = " ".join(f"{_expr(k)}:{_expr(v)}" for k, v in d.get("pairs", []))
        return f"{{{pairs}}}"

    if kind == "tuple_literal":
        elems = " ".join(_expr(e) for e in d.get("elements", []))
        return f"(tup {elems})"

    if kind == "lambda":
        params = " ".join(d.get("params", []))
        return f"(lambda [{params}] {_expr(d['body'])})"

    if kind == "conditional":
        return f'(? {_expr(d["cond"])} {_expr(d.get("then"))} {_expr(d.get("else"))})'

    if kind == "await":
        return f"(await {_expr(d['expr'])})"

    if kind == "cast":
        return f"(cast {_expr(d['expr'])} {_type_str(d.get('type', {}))})"

    if kind == "raw_expr":
        return f"(re {_q(d.get('text', ''))})"

    return f"(expr {_q(str(d))})"


def _stmt(d: Any) -> str:
    if not isinstance(d, dict):
        return str(d)

    kind = d.get("kind")

    if kind is None or kind == "block":
        stmts = d.get("stmts", [])
        if not stmts:
            return "(body)"
        inner = " ".join(_stmt(s) for s in stmts)
        return f"(body {inner})"

    if kind == "var_decl":
        name = d.get("name", "?")
        val = d.get("value")
        if val is not None:
            return f"(vd {name} {_expr(val)})"
        return f"(vd {name})"

    if kind == "assign":
        op = d.get("op", "=")
        return f"({op} {_expr(d['target'])} {_expr(d['value'])})"

    if kind == "return":
        val = d.get("value")
        return f"(return {_expr(val)})" if val is not None else "(return)"

    if kind == "if":
        then_key = "then" if "then" in d else "then_block"
        parts = ["if", _expr(d["cond"]), _stmt(d[then_key])]
        for cond, block in d.get("elif", []):
            parts.append(f"(elif {_expr(cond)} {_stmt(block)})")
        if "else" in d:
            parts.append(f"(else {_stmt(d['else'])})")
        return "(" + " ".join(parts) + ")"

    if kind == "while":
        return f"(while {_expr(d['cond'])} {_stmt(d['body'])})"

    if kind == "for_each":
        return f"(for {d['var']} {_expr(d['iter'])} {_stmt(d['body'])})"

    if kind == "match":
        arms = " ".join(
            f"({_q(a['pattern'])} {_stmt(a['body'])})"
            for a in d.get("arms", [])
        )
        return f"(match {_expr(d['subject'])} {arms})"

    if kind == "break":
        return "(break)"

    if kind == "continue":
        return "(continue)"

    if kind == "raise":
        val = d.get("expr")
        return f"(raise {_expr(val)})" if val is not None else "(raise)"

    if kind == "expr_stmt":
        return f"(es {_expr(d['expr'])})"

    if kind == "raw":
        return f"(raw {_q(d.get('text', ''))})"

    return f"(stmt {_q(str(d))})"


def _node(d: dict) -> str:
    kind = d.get("kind", "")

    if kind == "import":
        parts = ["imp", _attr("id", d.get("id", "")), _q(d.get("module", ""))]
        items = d.get("items")
        if items:
            parts.append("[" + " ".join(items) + "]")
        alias = d.get("alias")
        if alias:
            parts.append(_attr("as", alias))
        return "(" + " ".join(parts) + ")"

    if kind == "function":
        parts = ["fn", _attr("id", d.get("id", "")), d.get("name", "?")]
        params = d.get("params", [])
        if params:
            param_strs = []
            for p in params:
                p_name = p.get("name", "?")
                p_type = p.get("type")
                param_strs.append(f"{p_name}:{_type_str(p_type)}" if p_type else p_name)
            parts.append("[" + " ".join(param_strs) + "]")
        parts.append(_stmt(d.get("body", {})))
        return "(" + " ".join(parts) + ")"

    if kind == "variable":
        parts = ["var", _attr("id", d.get("id", "")), d.get("name", "?")]
        val = d.get("value")
        if val is not None:
            parts.append(_q(str(val)))
        return "(" + " ".join(parts) + ")"

    if kind == "type_def":
        cat = d.get("category", "class")
        parts = [cat, _attr("id", d.get("id", "")), d.get("name", "?")]
        fields = d.get("fields", [])
        methods = d.get("methods", [])
        if fields:
            field_strs = " ".join(
                f"(field {f.get('name', '?')}:{_type_str(f.get('type', {}))})"
                for f in fields
            )
            parts.append(f"(fields {field_strs})")
        if methods:
            methods_str = " ".join(_node(m) for m in methods)
            parts.append(f"(methods {methods_str})")
        return "(" + " ".join(parts) + ")"

    return f"(node {_q(str(d))})"


def sexpr(ast_dict: dict) -> str:
    """Serialize an AST dict to a multi-line S-expression string."""
    lang = ast_dict.get("source_language", "")
    file_ = ast_dict.get("source_file", "")
    lines = [f'(meta lang:{_q(lang)} file:{_q(file_)})']
    for node in ast_dict.get("nodes", []):
        lines.append(_node(node))
    return "\n".join(lines)
