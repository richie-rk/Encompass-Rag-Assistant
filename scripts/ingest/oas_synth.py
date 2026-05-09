"""Render an endpoint record's OAS spec (and optional `doc_api`) into markdown.

Endpoint pages on Developer Connect carry their docs as structured OpenAPI in the
`oas` field of each JSONL record, with `body_md` empty for ~95% of endpoints.
The retrieval pipeline can't embed structured JSON, so we render it to LLM-friendly
markdown here.

Public entry point:
    synthesize_endpoint_markdown(record) -> str

The renderer:
  - dereferences `$ref`s against `components.parameters` / `components.schemas`
    (depth-capped, cycle-guarded);
  - inlines schema JSON when small, falls back to a field table when large;
  - truncates oversized example payloads;
  - appends rendered code samples from `doc.api` when ReadMe has them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Tunables. Defaults chosen so a typical endpoint synthesizes to ~2-8K chars,
# which sits cleanly under the 2K-page-wise / 1500-char-split chunking thresholds.
MAX_DEREF_DEPTH = 4
MAX_INLINE_SCHEMA_CHARS = 4000      # inline as JSON if rendered size <= this
MAX_SCHEMA_TABLE_CHARS = 3500       # hard cap on field-table output per schema
MAX_EXAMPLE_CHARS = 4000
MAX_EXAMPLES_PER_BLOCK = 1  # avoid Model+Live duplication on this site
MAX_FIELD_TABLE_DEPTH = 1           # don't recurse into nested objects; field list only


# --- $ref resolution ----------------------------------------------------------

def _resolve_ref(ref: str, oas: Dict) -> Optional[Dict]:
    """Resolve a JSON-pointer-ish `#/components/<bucket>/<name>` against the OAS root."""
    if not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    cursor: Any = oas
    for p in parts:
        # OpenAPI uses `~1` for `/` and `~0` for `~` in JSON pointers.
        p = p.replace("~1", "/").replace("~0", "~")
        if not isinstance(cursor, dict) or p not in cursor:
            return None
        cursor = cursor[p]
    return cursor if isinstance(cursor, dict) else None


def _deref(node: Any, oas: Dict, *, depth: int = 0, seen: Optional[set] = None) -> Any:
    """Recursively replace `$ref` nodes with their resolved target.

    Returns a new structure; does not mutate the input. Cycles and over-deep
    references degrade to the unresolved `$ref` dict so the caller can still
    show *something*.
    """
    if depth >= MAX_DEREF_DEPTH:
        return node
    if seen is None:
        seen = set()

    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in seen:
                return {"$ref": ref, "_cycle": True}
            target = _resolve_ref(ref, oas)
            if target is None:
                return node
            return _deref(target, oas, depth=depth + 1, seen=seen | {ref})
        return {k: _deref(v, oas, depth=depth, seen=seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_deref(v, oas, depth=depth, seen=seen) for v in node]
    return node


# --- Example formatting -------------------------------------------------------

def _truncate_example(value: Any) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= MAX_EXAMPLE_CHARS:
        return text
    elided = len(text) - MAX_EXAMPLE_CHARS
    return text[:MAX_EXAMPLE_CHARS] + f"\n... (truncated, {elided:,} more chars)"


def _render_examples(examples: Optional[Dict], heading_level: int) -> List[str]:
    """OpenAPI `examples` is `{name: {value, summary?, description?}}`.

    Caps at `MAX_EXAMPLES_PER_BLOCK` per content block. ReadMe authors
    typically include redundant `Model Data Sample` + `Live Data Sample`
    pairs that differ only in mock data — keeping just one preserves
    retrieval value without doubling output size on every response.
    """
    if not isinstance(examples, dict) or not examples:
        return []
    h = "#" * heading_level
    out: List[str] = []
    rendered = 0
    extra = 0
    for name, ex in examples.items():
        if not isinstance(ex, dict):
            continue
        if rendered >= MAX_EXAMPLES_PER_BLOCK:
            extra += 1
            continue
        summary = ex.get("summary")
        title = name if not summary else f"{name} — {summary}"
        out.append(f"{h} {title}")
        if ex.get("description"):
            out.append(ex["description"])
        if "value" in ex:
            out.append("```json")
            out.append(_truncate_example(ex["value"]))
            out.append("```")
        rendered += 1
    if extra:
        out.append(f"_({extra} additional example(s) omitted)_")
    return out


# --- Schema rendering ---------------------------------------------------------

def _render_schema(schema: Optional[Dict], oas: Dict) -> List[str]:
    """Render a (already dereferenced) schema as either inline JSON or a field table."""
    if not schema:
        return []
    inline = json.dumps(schema, indent=2, ensure_ascii=False, default=str)
    if len(inline) <= MAX_INLINE_SCHEMA_CHARS:
        return ["```json", inline, "```"]
    return _render_schema_table(schema)


def _iter_properties(schema: Dict) -> Iterable[Tuple[str, Dict, bool]]:
    """Yield (name, prop_schema, required) for object-shaped schemas."""
    if not isinstance(schema, dict):
        return
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for name, prop in props.items():
        if isinstance(prop, dict):
            yield name, prop, name in required


def _type_label(schema: Dict) -> str:
    if not isinstance(schema, dict):
        return ""
    t = schema.get("type")
    if t == "array":
        items = schema.get("items") or {}
        inner = _type_label(items) or "object"
        return f"array<{inner}>"
    if isinstance(t, list):
        return "|".join(t)
    fmt = schema.get("format")
    if t and fmt:
        return f"{t} ({fmt})"
    if t:
        return str(t)
    if "oneOf" in schema or "anyOf" in schema:
        return "oneOf"
    if "allOf" in schema:
        return "allOf"
    return ""


def _flatten_description(desc: Optional[str]) -> str:
    """Markdown table cells can't contain raw newlines or pipes. Squash both."""
    if not desc:
        return ""
    return desc.replace("|", "\\|").replace("\n", " ").strip()


def _render_schema_table(schema: Dict, depth: int = 0) -> List[str]:
    """Field-table representation for schemas too large to inline.

    Hard-caps total output at `MAX_SCHEMA_TABLE_CHARS` to keep large schemas
    (e.g., `applicationnew` with hundreds of nested objects) from overwhelming
    the synthesized markdown. Past the cap, truncates with a note.
    """
    out: List[str] = []
    obj_type = schema.get("type") if isinstance(schema, dict) else None

    if obj_type == "array" and isinstance(schema.get("items"), dict):
        out.append("_Array of:_")
        out.extend(_render_schema_table(schema["items"], depth=depth))
        return out

    rows = list(_iter_properties(schema))
    if not rows:
        label = _type_label(schema) or "schema"
        if schema.get("description"):
            return [f"_{label}_ — {_flatten_description(schema['description'])}"]
        return [f"_{label}_"]

    out.append("| Field | Type | Required | Description |")
    out.append("| --- | --- | --- | --- |")

    running = sum(len(s) + 1 for s in out)
    truncated = 0
    emitted = 0
    for name, prop, required in rows:
        type_str = _type_label(prop)
        desc = _flatten_description(prop.get("description"))
        row = f"| `{name}` | {type_str} | {'yes' if required else 'no'} | {desc} |"
        if running + len(row) + 1 > MAX_SCHEMA_TABLE_CHARS:
            truncated = len(rows) - emitted
            break
        out.append(row)
        running += len(row) + 1
        emitted += 1

    if truncated:
        out.append(f"_({truncated} more field(s) omitted; see full schema in OAS)_")
    return out


# --- Parameters ---------------------------------------------------------------

def _render_parameters(parameters: List[Dict], oas: Dict) -> List[str]:
    if not parameters:
        return []
    out = ["### Parameters", "", "| Name | In | Required | Type | Description |",
           "| --- | --- | --- | --- | --- |"]
    for p in parameters:
        if not isinstance(p, dict):
            continue
        deref = _deref(p, oas)
        if not isinstance(deref, dict):
            continue
        name = deref.get("name") or ""
        loc = deref.get("in") or ""
        required = deref.get("required", False)
        schema = deref.get("schema") or {}
        type_str = _type_label(schema) if isinstance(schema, dict) else ""
        desc = _flatten_description(deref.get("description"))
        out.append(f"| `{name}` | {loc} | {'yes' if required else 'no'} | {type_str} | {desc} |")
    return out


# --- Request / response bodies ------------------------------------------------

def _render_content_block(
    label: str,
    content: Dict,
    oas: Dict,
    *,
    schema_heading_level: int = 4,
    example_heading_level: int = 5,
) -> List[str]:
    """Render the `content` map of a requestBody or response (keyed by content-type)."""
    if not isinstance(content, dict) or not content:
        return []
    out: List[str] = []
    for ct, body in content.items():
        if not isinstance(body, dict):
            continue
        out.append(f"Content-Type: `{ct}`")
        out.append("")
        schema = _deref(body.get("schema") or {}, oas)
        rendered = _render_schema(schema, oas)
        if rendered:
            sh = "#" * schema_heading_level
            out.append(f"{sh} Schema")
            out.extend(rendered)
            out.append("")
        examples = body.get("examples") or {}
        if examples:
            sh = "#" * schema_heading_level
            out.append(f"{sh} Examples")
            out.extend(_render_examples(examples, example_heading_level))
            out.append("")
        elif "example" in body:
            sh = "#" * schema_heading_level
            out.append(f"{sh} Example")
            out.append("```json")
            out.append(_truncate_example(body["example"]))
            out.append("```")
            out.append("")
    return out


# --- doc.api code samples -----------------------------------------------------

def _render_doc_api_codes(doc_api: Optional[Dict]) -> List[str]:
    """Pull rendered cURL / Python / JS examples out of `doc.api` when present.

    Shape varies per page: `examples.codes`, `results.codes`, sometimes both.
    Each entry is `{language, code, name?, status?}`.
    """
    if not isinstance(doc_api, dict):
        return []
    code_blocks: List[Tuple[str, str, str]] = []  # (heading, lang, code)

    def _is_meaningful(code: str) -> bool:
        """Skip ReadMe stub responses like `{}`, `[]`, `null` — empty placeholders
        that the API explorer leaves behind on pages without real interactions."""
        s = (code or "").strip()
        if len(s) < 5:
            return False
        return s not in ("{}", "[]", "null", '""', "''")

    examples = (doc_api.get("examples") or {}).get("codes") or []
    for c in examples:
        if not isinstance(c, dict):
            continue
        lang = (c.get("language") or "").lower() or "text"
        code = c.get("code") or ""
        if not _is_meaningful(code):
            continue
        name = c.get("name") or lang
        code_blocks.append((f"{name}", lang, code))

    results = (doc_api.get("results") or {}).get("codes") or []
    for c in results:
        if not isinstance(c, dict):
            continue
        lang = (c.get("language") or "").lower() or "text"
        code = c.get("code") or ""
        if not _is_meaningful(code):
            continue
        status = c.get("status")
        name_bits = []
        if status is not None:
            name_bits.append(f"Response {status}")
        name_bits.append(c.get("name") or lang)
        code_blocks.append((" — ".join(name_bits), lang, code))

    if not code_blocks:
        return []

    out = ["### Code samples (rendered by ReadMe)", ""]
    for heading, lang, code in code_blocks:
        out.append(f"#### {heading}")
        out.append(f"```{lang}")
        out.append(code if len(code) <= MAX_EXAMPLE_CHARS
                   else code[:MAX_EXAMPLE_CHARS] + f"\n... (truncated)")
        out.append("```")
        out.append("")
    return out


# --- Operation rendering ------------------------------------------------------

_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options", "trace")


def _render_operation(method: str, path: str, op: Dict, oas: Dict) -> List[str]:
    out = [f"## {method.upper()} {path}", ""]
    summary = op.get("summary")
    description = op.get("description")
    if summary:
        out.append(f"**{summary}**")
        out.append("")
    if description:
        out.append(str(description).strip())
        out.append("")
    if op.get("deprecated"):
        out.append("> ⚠ This operation is marked deprecated.")
        out.append("")

    parameters = op.get("parameters") or []
    if parameters:
        out.extend(_render_parameters(parameters, oas))
        out.append("")

    request_body = op.get("requestBody")
    if isinstance(request_body, dict):
        rb = _deref(request_body, oas)
        if isinstance(rb, dict) and rb.get("content"):
            out.append("### Request body")
            out.append("")
            if rb.get("required"):
                out.append("_Required._")
                out.append("")
            if rb.get("description"):
                out.append(str(rb["description"]).strip())
                out.append("")
            out.extend(_render_content_block("request body", rb["content"], oas))

    responses = op.get("responses") or {}
    if responses:
        out.append("### Responses")
        out.append("")
        for code, resp in responses.items():
            if not isinstance(resp, dict):
                continue
            resp_d = _deref(resp, oas)
            if not isinstance(resp_d, dict):
                continue
            desc = (resp_d.get("description") or "").strip()
            head = f"#### {code}"
            if desc:
                head = f"{head} — {desc}"
            out.append(head)
            out.append("")
            # 4xx/5xx responses on this site reuse a generic Error schema across
            # every operation — rendering it dozens of times bloats synthesis
            # without retrieval value. Render schemas/examples for 2xx only.
            is_success = isinstance(code, str) and code.startswith("2")
            if is_success and resp_d.get("content"):
                out.extend(_render_content_block(
                    "response", resp_d["content"], oas,
                    schema_heading_level=5, example_heading_level=6,
                ))

    return out


# --- Public API ---------------------------------------------------------------

def synthesize_endpoint_markdown(record: Dict) -> str:
    """Render the OAS+doc.api content of a single JSONL record into markdown.

    Returns an empty string when the record has nothing renderable (no OAS paths
    AND no usable doc.api code samples).
    """
    oas = record.get("oas") or {}
    paths = oas.get("paths") or {}
    parts: List[str] = []

    for path, ops in paths.items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in _HTTP_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            parts.extend(_render_operation(method, path, op, oas))
            parts.append("")

    code_samples_md = _render_doc_api_codes(record.get("doc_api"))
    if code_samples_md:
        parts.extend(code_samples_md)

    return "\n".join(parts).rstrip() + ("\n" if parts else "")
