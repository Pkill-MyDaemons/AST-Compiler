# ast-compiler

Language-agnostic AST compiler and decompiler — parse source files into a unified JSON AST, optimize, edit, and compile back to source.

## Features

- **Decompile** Python, Rust, TypeScript/TSX source to a unified JSON AST
- **Compile** a unified AST back to source in any supported language
- **Three output formats** — full verbose JSON, compressed min-JSON for LLM token efficiency, and S-expression for LLM token efficiency
- **Optimization passes** — constant folding, identity elimination, dead code elimination
- **AI harness** (`ast-harness`) for programmatic AST inspection and editing: rename nodes, update types, add/remove methods, splice statements
- **Language auto-detection** from file extension
- **Skeleton view** — names and signatures only, no bodies, for low-cost LLM context

## Supported Languages

| Language | Aliases | Status |
|---|---|---|
| Python | `py` | Full parser + generator |
| Rust | `rs` | Full parser + generator |
| TypeScript | `ts`, `mts`, `cts` | Full parser + generator |
| TSX | `tsx` | Full parser + generator |
| Go | `go` | Skeleton (tree-sitter, needs package) |
| JavaScript | `js`, `mjs`, `cjs` | Skeleton (tree-sitter, needs package) |
| Java | `java` | Skeleton (tree-sitter, needs package) |
| Kotlin | `kt`, `kts` | Skeleton (tree-sitter, needs package) |
| Swift | `swift` | Skeleton (tree-sitter, needs package) |
| Zig | `zig` | Skeleton (tree-sitter, needs package) |
| C | `c`, `h` | Skeleton (tree-sitter, needs package) |
| C++ | `cpp`, `cc`, `cxx`, `hpp`, `hxx` | Skeleton (tree-sitter, needs package) |

Skeleton parsers require the corresponding `tree-sitter-*` Python package to be installed and are not yet wired to a code generator.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs two commands: `ast-compiler` and `ast-harness`.

## Quick Start

### Decompile source to AST

```bash
# Auto-detect language from extension, write to stdout
ast-compiler decompile src/main.py

# Write to a file
ast-compiler decompile src/main.py -o main.ast.json

# Override language detection
ast-compiler decompile src/lib --lang rust -o lib.ast.json
```

### Compile AST back to source

```bash
ast-compiler compile main.ast.json --lang python
ast-compiler compile main.ast.json --lang rust -o out.rs
```

### Run optimization passes on an AST file

```bash
# All passes (overwrites input)
ast-compiler optimize main.ast.json

# Write to a new file
ast-compiler optimize main.ast.json -o main.opt.ast.json

# Select specific passes
ast-compiler optimize main.ast.json --passes fold,dce
```

### Inspect an AST file

```bash
ast-compiler info main.ast.json
```

Example output:

```
Language : python
File     : main.py
Version  : 1
Nodes    : 12 top-level
  function: 8
  import: 3
  type_def: 1
```

## Output Formats

Pass `--format` to `decompile` to control output. All three formats represent the same AST; they differ in verbosity.

### `verbose` (default)

Full pretty-printed JSON. Human-readable; largest token count.

```bash
ast-compiler decompile add.py --format verbose
```

```json
{
  "source_language": "python",
  "source_file": "add.py",
  "version": 1,
  "nodes": [
    {
      "kind": "function",
      "id": "fn_add",
      "name": "add",
      "params": [
        { "name": "x", "type": { "kind": "number", "bits": 64 } },
        { "name": "y", "type": { "kind": "number", "bits": 64 } }
      ],
      "body": {
        "stmts": [
          { "kind": "return", "value": { "kind": "binary_op", "op": "+", "left": { "kind": "identifier", "name": "x" }, "right": { "kind": "identifier", "name": "y" } } }
        ]
      }
    }
  ]
}
```

### `min-json`

Compressed JSON — keys and common kind values are shortened. Designed to reduce LLM context usage.

```bash
ast-compiler decompile add.py --format min-json
```

```json
{"lang":"python","file":"add.py","nodes":[{"k":"fn","i":"fn_add","v":"add","params":[{"v":"x","type":{"kind":"number","bits":64}},{"v":"y","type":{"kind":"number","bits":64}}],"b":{"s":[{"kind":"return","value":{"kind":"binary_op","op":"+","left":{"kind":"identifier","name":"x"},"right":{"kind":"identifier","name":"y"}}}]}}]}
```

Key compression map: `kind`→`k`, `id`→`i`, `body`→`b`, `stmts`→`s`, `expr`→`e`, `func`→`f`, `args`→`a`, `text`/`name`/`value`→`v`. Common kind values: `function`→`fn`, `import`→`imp`, `expr_stmt`→`es`, `var_decl`→`vd`, `field_access`→`fa`.

### `sexpr`

Lisp-style S-expressions. Most compact; suited for LLMs comfortable with structured symbolic notation.

```bash
ast-compiler decompile add.py --format sexpr
```

```
(meta lang:"python" file:"add.py")
(fn id:"fn_add" add [x:num64 y:num64] (body (return (+ x y))))
```

## Optimization Passes

The `optimize` command (and `decompile`, which runs optimization automatically) applies these passes:

| Pass | Flag | Description |
|---|---|---|
| Constant folding | `fold` | Evaluates constant arithmetic and boolean expressions at compile time |
| Identity elimination | `identity` | Removes identity operations (`x + 0`, `x * 1`, `x or False`, etc.) |
| Dead code elimination | `dce` | Removes unreachable statements after unconditional `return`/`break`/`continue` |

Run all passes:

```bash
ast-compiler optimize program.ast.json
```

Run a subset:

```bash
ast-compiler optimize program.ast.json --passes fold,identity
ast-compiler optimize program.ast.json --passes dce
```

## AI Harness (`ast-harness`)

`ast-harness` provides targeted read and edit commands for working with AST files in an agentic loop. All edit commands write changes back to the AST file in place.

JSON arguments can be passed inline or from a file using the `@path` prefix (e.g., `@stmt.json`).

### `skeleton` — list names and signatures

```bash
ast-harness skeleton program.ast.json
```

Returns all top-level node names, parameter lists, and return types — no function bodies. Use this to orient an LLM before reading individual nodes.

### `get` — fetch a single node

```bash
ast-harness get program.ast.json fn_process_data
```

Returns the full JSON for that node including its body.

### `str-replace` — replace a statement in a function body

```bash
ast-harness str-replace program.ast.json fn_process_data \
  '{"kind":"return","value":{"kind":"literal","value":0}}' \
  '{"kind":"return","value":{"kind":"literal","value":1}}'
```

Finds the first statement in `fn_process_data` that matches the old JSON and replaces it.

### `rename` — rename a node

```bash
ast-harness rename program.ast.json fn_old_name new_name
```

### `set-return-type` — change a function's return type

```bash
ast-harness set-return-type program.ast.json fn_compute '{"kind":"number","bits":64}'
```

### `set-param-type` — change a parameter's type

```bash
ast-harness set-param-type program.ast.json fn_compute value '{"kind":"string"}'
```

### `add-method` — add a method to a type

```bash
ast-harness add-method program.ast.json type_MyClass @new_method.json
```

`new_method.json` must be a valid function node dict.

### `remove` — remove a node

```bash
ast-harness remove program.ast.json fn_unused_helper
```

### `append-stmt` — append a statement to a function body

```bash
ast-harness append-stmt program.ast.json fn_init \
  '{"kind":"expr_stmt","expr":{"kind":"call","func":{"kind":"identifier","name":"setup"},"args":[]}}'
```

### `insert-before` — insert a statement at a position

```bash
# Prepend (index 0)
ast-harness insert-before program.ast.json fn_init 0 @log_stmt.json

# Insert at position 2
ast-harness insert-before program.ast.json fn_init 2 @log_stmt.json
```

## Running Tests

```bash
pytest
```

Test coverage includes roundtrip parsing (decompile → compile → decompile), optimizer correctness, cross-language AST equivalence, and the `min-json`/`sexpr` token-thrift formats.

## License

See [LICENSE](LICENSE).
