"""CLI entrypoints: ast-compiler and ast-harness."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import click

from .parsers import parse, detect_language
from .generators import generate
from .unified_ast.nodes import Module
from .optimizer import optimize
from .harness.skeleton import build_skeleton
from .harness.editor import (
    get_node, str_replace_body, rename_node,
    update_return_type, update_param,
    add_method, add_field, remove_node, append_stmt, insert_stmt_at,
    save, load,
)


# ===========================================================================
# ast-compiler
# ===========================================================================

@click.group("ast-compiler")
def compiler_cli() -> None:
    """Decompile source code to unified AST JSON, or compile AST back to source."""


@compiler_cli.command("decompile")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output JSON file (default: stdout)")
@click.option("--lang", default=None, help="Override language detection (python|rust)")
@click.option("--pretty/--no-pretty", default=True, help="Pretty-print JSON")
@click.option(
    "--format", "fmt",
    default="verbose",
    type=click.Choice(["verbose", "min-json", "sexpr"]),
    help="Output format: verbose (default), min-json, or sexpr",
)
def cmd_decompile(input_file: str, output: str, lang: str, pretty: bool, fmt: str) -> None:
    """Parse source code into unified AST JSON."""
    path = Path(input_file)
    source = path.read_text(encoding="utf-8")
    language = lang or detect_language(path.name)

    try:
        module = parse(source, language, filename=path.name)
        module = optimize(module)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    ast_dict = module.to_dict()

    if fmt == "min-json":
        from .unified_ast.minify import minify
        result = minify(ast_dict)
    elif fmt == "sexpr":
        from .unified_ast.sexpr import sexpr
        result = sexpr(ast_dict)
    else:
        indent = 2 if pretty else None
        result = json.dumps(ast_dict, indent=indent, ensure_ascii=False)

    if output:
        Path(output).write_text(result + "\n", encoding="utf-8")
        click.echo(f"Wrote AST to {output}", err=True)
    else:
        click.echo(result)


@compiler_cli.command("compile")
@click.argument("ast_file", type=click.Path(exists=True))
@click.option("--lang", required=True, help="Target language (python|rust)")
@click.option("-o", "--output", default=None, help="Output source file (default: stdout)")
def cmd_compile(ast_file: str, lang: str, output: str) -> None:
    """Compile unified AST JSON back to source code."""
    try:
        module = load(ast_file)
        module = optimize(module)
        source = generate(module, lang)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output:
        Path(output).write_text(source, encoding="utf-8")
        click.echo(f"Wrote {lang} source to {output}", err=True)
    else:
        click.echo(source)


@compiler_cli.command("optimize")
@click.argument("ast_file", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output JSON file (default: overwrite input)")
@click.option("--pretty/--no-pretty", default=True, help="Pretty-print JSON")
@click.option("--passes", default=None,
              help="Comma-separated list of passes: fold,identity,dce (default: all)")
def cmd_optimize(ast_file: str, output: str, pretty: bool, passes: str) -> None:
    """Run optimization passes on an AST JSON file."""
    from .optimizer import ConstantFoldingPass, IdentityEliminationPass, DeadCodeEliminationPass
    _ALL = {
        "fold": ConstantFoldingPass,
        "identity": IdentityEliminationPass,
        "dce": DeadCodeEliminationPass,
    }
    try:
        module = load(ast_file)
    except Exception as e:
        click.echo(f"Error loading {ast_file}: {e}", err=True)
        sys.exit(1)

    if passes:
        selected = [_ALL[p.strip()]() for p in passes.split(",") if p.strip() in _ALL]
    else:
        selected = None  # all

    before = len(json.dumps(module.to_dict()))
    module = optimize(module, selected)
    after = len(json.dumps(module.to_dict()))

    indent = 2 if pretty else None
    result = json.dumps(module.to_dict(), indent=indent, ensure_ascii=False)
    dest = output or ast_file
    import pathlib
    pathlib.Path(dest).write_text(result + "\n", encoding="utf-8")
    click.echo(f"Optimized {ast_file} → {dest}  ({before} → {after} chars, "
               f"{100*(before-after)//before}% smaller)", err=True)


@compiler_cli.command("info")
@click.argument("ast_file", type=click.Path(exists=True))
def cmd_info(ast_file: str) -> None:
    """Print a summary of an AST file."""
    try:
        module = load(ast_file)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    counts: dict = {}
    for n in module.nodes:
        counts[n.kind] = counts.get(n.kind, 0) + 1
        if n.kind == "type_def":
            for m in n.methods:  # type: ignore[union-attr]
                counts["function"] = counts.get("function", 0) + 1

    click.echo(f"Language : {module.source_language}")
    click.echo(f"File     : {module.source_file}")
    click.echo(f"Version  : {module.version}")
    click.echo(f"Nodes    : {len(module.nodes)} top-level")
    for kind, count in sorted(counts.items()):
        click.echo(f"  {kind}: {count}")


# ===========================================================================
# ast-harness
# ===========================================================================

@click.group("ast-harness")
def harness_cli() -> None:
    """AI harness: inspect and edit unified AST files."""


@harness_cli.command("skeleton")
@click.argument("ast_file", type=click.Path(exists=True))
@click.option("--pretty/--no-pretty", default=True)
def cmd_skeleton(ast_file: str, pretty: bool) -> None:
    """Print the skeleton (names and signatures only — no bodies)."""
    module = load(ast_file)
    skel = build_skeleton(module)
    indent = 2 if pretty else None
    click.echo(json.dumps(skel, indent=indent, ensure_ascii=False))


@harness_cli.command("get")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("node_id")
@click.option("--pretty/--no-pretty", default=True)
def cmd_get(ast_file: str, node_id: str, pretty: bool) -> None:
    """Print the full content of a node (including body)."""
    try:
        module = load(ast_file)
        node = get_node(module, node_id)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    indent = 2 if pretty else None
    click.echo(json.dumps(node, indent=indent, ensure_ascii=False))


@harness_cli.command("str-replace")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("node_id")
@click.argument("old_stmt_json")
@click.argument("new_stmt_json")
def cmd_str_replace(ast_file: str, node_id: str, old_stmt_json: str, new_stmt_json: str) -> None:
    """Replace a statement in a function body.

    OLD_STMT_JSON and NEW_STMT_JSON must be JSON statement objects (or file paths prefixed with @).
    """
    old_json = _maybe_read_file(old_stmt_json)
    new_json = _maybe_read_file(new_stmt_json)
    try:
        module = load(ast_file)
        str_replace_body(module, node_id, old_json, new_json)
        save(module, ast_file)
        click.echo(f"Replaced statement in {node_id}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("rename")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("node_id")
@click.argument("new_name")
def cmd_rename(ast_file: str, node_id: str, new_name: str) -> None:
    """Rename a node (function, type, variable)."""
    try:
        module = load(ast_file)
        rename_node(module, node_id, new_name)
        save(module, ast_file)
        click.echo(f"Renamed {node_id} → {new_name}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("set-return-type")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("fn_id")
@click.argument("type_json")
def cmd_set_return_type(ast_file: str, fn_id: str, type_json: str) -> None:
    """Change the return type of a function. TYPE_JSON e.g. '{\"kind\":\"number\",\"bits\":64}'"""
    raw = _maybe_read_file(type_json)
    try:
        type_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid type JSON: {e}", err=True)
        sys.exit(1)
    try:
        module = load(ast_file)
        update_return_type(module, fn_id, type_dict)
        save(module, ast_file)
        click.echo(f"Updated return type of {fn_id}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("set-param-type")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("fn_id")
@click.argument("param_name")
@click.argument("type_json")
def cmd_set_param_type(ast_file: str, fn_id: str, param_name: str, type_json: str) -> None:
    """Change the type of a function parameter."""
    raw = _maybe_read_file(type_json)
    try:
        type_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid type JSON: {e}", err=True)
        sys.exit(1)
    try:
        module = load(ast_file)
        update_param(module, fn_id, param_name, type_dict)
        save(module, ast_file)
        click.echo(f"Updated type of {fn_id} param {param_name!r}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("add-method")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("type_id")
@click.argument("fn_json")
def cmd_add_method(ast_file: str, type_id: str, fn_json: str) -> None:
    """Add a new method to a type_def. FN_JSON is a function node dict or @file."""
    raw = _maybe_read_file(fn_json)
    try:
        fn_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid function JSON: {e}", err=True)
        sys.exit(1)
    try:
        module = load(ast_file)
        add_method(module, type_id, fn_dict)
        save(module, ast_file)
        click.echo(f"Added method to {type_id}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("remove")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("node_id")
def cmd_remove(ast_file: str, node_id: str) -> None:
    """Remove a node from the AST."""
    try:
        module = load(ast_file)
        remove_node(module, node_id)
        save(module, ast_file)
        click.echo(f"Removed {node_id}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("append-stmt")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("fn_id")
@click.argument("stmt_json")
def cmd_append_stmt(ast_file: str, fn_id: str, stmt_json: str) -> None:
    """Append a statement to the end of a function body."""
    raw = _maybe_read_file(stmt_json)
    try:
        stmt_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid stmt JSON: {e}", err=True)
        sys.exit(1)
    try:
        module = load(ast_file)
        append_stmt(module, fn_id, stmt_dict)
        save(module, ast_file)
        click.echo(f"Appended statement to {fn_id}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@harness_cli.command("insert-before")
@click.argument("ast_file", type=click.Path(exists=True))
@click.argument("fn_id")
@click.argument("index", type=int)
@click.argument("stmt_json")
def cmd_insert_before(ast_file: str, fn_id: str, index: int, stmt_json: str) -> None:
    """Insert a statement at position INDEX in a function body (0 = prepend)."""
    raw = _maybe_read_file(stmt_json)
    try:
        stmt_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid stmt JSON: {e}", err=True)
        sys.exit(1)
    try:
        module = load(ast_file)
        insert_stmt_at(module, fn_id, index, stmt_dict)
        save(module, ast_file)
        click.echo(f"Inserted statement at index {index} in {fn_id}", err=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _maybe_read_file(s: str) -> str:
    """If s starts with @, treat it as a file path and return its contents."""
    if s.startswith("@"):
        return Path(s[1:]).read_text(encoding="utf-8")
    return s
