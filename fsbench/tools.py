"""Agent-facing toolsets over a sandboxed workspace, with full I/O tracing.

Three retrieval interfaces over the same files:

``shell``    ls / cd / pwd / find / grep / cat / convert_to_text. grep scans raw bytes, so it
             cannot see inside PDF/XLSX/DOCX, and cat refuses binary files, as on a real box.
``files``    list_directory / read_file / glob / search_files. read_file and search_files
             understand every format.
``indexed``  ``files`` plus search_index, a ranked BM25 index over names and contents.

All toolsets expose write_file unless writes are disabled. Paths are case-sensitive and
virtualised under /workspace regardless of host OS, so traces match the manifest exactly.
"""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fsbench.extract import extract_text, is_binary
from fsbench.paths import long_path

VROOT = "/workspace"


class ToolError(Exception):
    pass


class Tracer:
    """Collects one event per tool call; optionally streams them to a JSONL file."""

    def __init__(self, path: str | Path | None = None):
        self.events: list[dict] = []
        self._fh = open(long_path(path), "w", encoding="utf-8") if path else None
        self._t0 = time.perf_counter()

    def log(self, event: dict) -> None:
        event = {"step": len(self.events) + 1, "t": round(time.perf_counter() - self._t0, 4), **event}
        self.events.append(event)
        if self._fh:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _param(type_: str, description: str, default: Any = ..., **extra) -> dict:
    p = {"type": type_, "description": description, **extra}
    if default is not ...:
        p["default"] = default
    return p


SPECS: dict[str, tuple[str, dict[str, dict], list[str]]] = {
    "ls": ("List a directory (directories end with '/').",
           {"path": _param("string", "Directory or file path.", "."),
            "long": _param("boolean", "Also show file sizes in bytes.", False)}, []),
    "cd": ("Change the working directory.", {"path": _param("string", "Directory path.")}, ["path"]),
    "pwd": ("Print the working directory.", {}, []),
    "find": ("Recursively find paths under a directory, like `find PATH -name GLOB -type f|d`.",
             {"path": _param("string", "Directory to search.", "."),
              "name": _param("string", "Case-sensitive glob on the basename, e.g. '*.csv'."),
              "type": _param("string", "'f' for files, 'd' for directories.", enum=["f", "d"]),
              "maxdepth": _param("integer", "Maximum depth below path.")}, []),
    "grep": ("Search file bytes for a regular expression, like `grep -rn`. Does not decode PDF/XLSX/DOCX.",
             {"pattern": _param("string", "Python regular expression."),
              "path": _param("string", "File or directory.", "."),
              "ignore_case": _param("boolean", "Case-insensitive match.", False),
              "files_only": _param("boolean", "Only print matching file paths, like -l.", False)}, ["pattern"]),
    "cat": ("Print a text file.", {"path": _param("string", "File path.")}, ["path"]),
    "convert_to_text": ("Extract text from a PDF, XLSX, DOCX, or text file (like pdftotext / xlsx2csv).",
                        {"path": _param("string", "File path.")}, ["path"]),
    "list_directory": ("List the entries of a directory (directories end with '/').",
                       {"path": _param("string", "Directory path.", VROOT)}, []),
    "read_file": ("Read any file as text. PDF, XLSX and DOCX are converted automatically.",
                  {"path": _param("string", "File path.")}, ["path"]),
    "glob": ("Find files whose path matches a glob pattern relative to path, e.g. '**/*.pdf'.",
             {"pattern": _param("string", "Glob pattern; ** matches any number of folders."),
              "path": _param("string", "Base directory.", VROOT)}, ["pattern"]),
    "search_files": ("Case-insensitive substring search over file names and contents (all formats).",
                     {"query": _param("string", "Text to look for."),
                      "path": _param("string", "Directory to search.", VROOT),
                      "max_results": _param("integer", "Maximum files to return.", 50)}, ["query"]),
    "search_index": ("Ranked keyword search (BM25) over the indexed contents and names of all files.",
                     {"query": _param("string", "Natural-language or keyword query."),
                      "k": _param("integer", "Number of results.", 10)}, ["query"]),
    "write_file": ("Create or overwrite a text file, creating parent folders as needed.",
                   {"path": _param("string", "File path."), "content": _param("string", "Full file contents.")},
                   ["path", "content"]),
}

TOOL_CATEGORY = {
    "ls": "navigate", "cd": "navigate", "pwd": "navigate", "find": "navigate",
    "list_directory": "navigate", "glob": "navigate",
    "grep": "search", "search_files": "search", "search_index": "search",
    "cat": "read", "convert_to_text": "read", "read_file": "read",
    "write_file": "write",
}


class ToolSet:
    name = "base"
    tool_names: tuple[str, ...] = ()

    def __init__(
        self,
        root: str | Path,
        tracer: Tracer | None = None,
        *,
        allow_writes: bool = True,
        max_output_chars: int = 20000,
    ):
        self.root = long_path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(root)
        self.tracer = tracer or Tracer()
        self.allow_writes = allow_writes
        self.max_output_chars = max_output_chars
        self.cwd = VROOT
        self._text_cache: dict[str, tuple[tuple[int, int], str]] = {}
        self._cur: dict[str, list] = {}

    @property
    def available(self) -> tuple[str, ...]:
        return self.tool_names + (("write_file",) if self.allow_writes else ())

    def specs(self, style: str = "anthropic") -> list[dict]:
        out = []
        for name in self.available:
            desc, props, required = SPECS[name]
            schema = {"type": "object", "properties": props, "required": required}
            if style == "anthropic":
                out.append({"name": name, "description": desc, "input_schema": schema})
            elif style == "openai":
                out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
            else:
                raise ValueError(f"unknown spec style {style!r}")
        return out

    def call(self, name: str, args: dict | None = None) -> str:
        args = dict(args or {})
        self._cur = {"read": [], "seen": [], "missing": [], "written": [], "path_args": [0],
                     "parse_ms": [0.0], "index_build_ms": [0.0]}
        start = time.perf_counter()
        error = None
        try:
            if name not in self.available:
                raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.available)}")
            _, props, required = SPECS[name]
            unknown = sorted(set(args) - set(props))
            missing = [r for r in required if r not in args]
            if unknown or missing:
                raise ToolError(f"bad arguments for {name}: unknown={unknown} missing={missing}")
            text = getattr(self, f"_t_{name}")(**args)
        except ToolError as e:
            error, text = str(e), f"Error: {e}"
        truncated = len(text) > self.max_output_chars
        if truncated:
            text = text[: self.max_output_chars] + f"\n...[output truncated: {len(text) - self.max_output_chars} more characters]"
            # Record candidates actually delivered, not paths beyond the output cap.
            visible = text.split("\n...[output truncated:", 1)[0]
            complete_lines = visible.rsplit("\n", 1)[0] if "\n" in visible else ""
            if name in ("ls", "list_directory"):
                self._cur["seen"] = [p for p in self._cur["seen"]
                                     if any(line.strip() == posixpath.basename(p)
                                            or line.rstrip().endswith("  " + posixpath.basename(p))
                                            or line.strip() == self._v(p)
                                            for line in complete_lines.splitlines())]
            else:
                self._cur["seen"] = [p for p in self._cur["seen"]
                                     if re.search(re.escape(self._v(p)) + r"(?=$|[\s:])", complete_lines)]
        self.tracer.log({
            "tool": name,
            "category": TOOL_CATEGORY.get(name, "other"),
            "args": args,
            "cwd": self.cwd,
            "ok": error is None,
            "error": error,
            "read": _unique(self._cur["read"]),
            "seen": _unique(self._cur["seen"]),
            "missing": _unique(self._cur["missing"]),
            "written": _unique(self._cur["written"]),
            "path_args": self._cur["path_args"][0],
            "output_chars": len(text),
            "truncated": truncated,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "parse_ms": round(self._cur["parse_ms"][0], 2),
            "index_build_ms": round(self._cur["index_build_ms"][0], 2) or None,
        })
        return text

    # -- path handling -----------------------------------------------------

    def _resolve(self, path: str, *, must_exist: bool = True, kind: str | None = None) -> tuple[Path, str]:
        self._cur["path_args"][0] += 1
        raw = path
        p = (path or ".").strip().replace("\\", "/")
        if not p.startswith("/"):
            p = posixpath.join(self.cwd, p)
        p = posixpath.normpath(p)
        if p == VROOT:
            rel = ""
        elif p.startswith(VROOT + "/"):
            rel = p[len(VROOT) + 1:]
        else:
            self._cur["missing"].append(p)
            raise ToolError(f"{raw}: No such file or directory (only {VROOT} is accessible)")
        os_path = self.root.joinpath(*rel.split("/")) if rel else self.root
        if must_exist:
            if not self._exists_exact(rel):
                self._cur["missing"].append(p)
                raise ToolError(f"{raw}: No such file or directory")
            if kind == "dir" and not os_path.is_dir():
                raise ToolError(f"{raw}: Not a directory")
            if kind == "file" and not os_path.is_file():
                raise ToolError(f"{raw}: Is a directory")
        return os_path, rel

    def _exists_exact(self, rel: str) -> bool:
        cur = self.root
        for seg in rel.split("/") if rel else []:
            try:
                if seg not in os.listdir(cur):
                    return False
            except (FileNotFoundError, NotADirectoryError):
                return False
            cur = cur / seg
        return True

    @staticmethod
    def _v(rel: str) -> str:
        return f"{VROOT}/{rel}" if rel else VROOT

    def _walk(self, os_dir: Path, rel_dir: str, maxdepth: int | None = None):
        """Yield (rel_path, is_dir, depth) below a directory, sorted."""
        stack = [(os_dir, rel_dir, 0)]
        while stack:
            d, r, depth = stack.pop()
            try:
                names = sorted(os.listdir(d), reverse=True)
            except OSError:
                continue
            for name in names:
                child_rel = f"{r}/{name}" if r else name
                is_dir = (d / name).is_dir()
                yield child_rel, is_dir, depth + 1
                if is_dir and (maxdepth is None or depth + 1 < maxdepth):
                    stack.append((d / name, child_rel, depth + 1))

    def _all_files(self, os_dir: Path, rel_dir: str) -> list[str]:
        return sorted(r for r, is_dir, _ in self._walk(os_dir, rel_dir) if not is_dir)

    def _text(self, rel: str) -> str:
        os_path = self.root.joinpath(*rel.split("/"))
        st = os_path.stat()
        key = (st.st_mtime_ns, st.st_size)
        hit = self._text_cache.get(rel)
        if hit and hit[0] == key:
            return hit[1]
        t0 = time.perf_counter()
        try:
            text = extract_text(os_path)
        except Exception as e:
            raise ToolError(f"{self._v(rel)}: could not extract text ({e})")
        self._cur.setdefault("parse_ms", [0.0])[0] += (time.perf_counter() - t0) * 1000
        self._text_cache[rel] = (key, text)
        return text

    def _list(self, path: str, long: bool = False) -> str:
        os_path, rel = self._resolve(path)
        if os_path.is_file():
            self._cur["seen"].append(rel)
            return self._v(rel)
        lines = []
        for name in sorted(os.listdir(os_path)):
            child = os_path / name
            child_rel = f"{rel}/{name}" if rel else name
            if child.is_dir():
                lines.append(f"{name}/")
            else:
                self._cur["seen"].append(child_rel)
                lines.append(f"{child.stat().st_size:>10}  {name}" if long else name)
        return "\n".join(lines) if lines else "(empty directory)"

    # -- shared tools --------------------------------------------------------

    def _t_write_file(self, path: str, content: str) -> str:
        if not self.allow_writes:
            raise ToolError("writes are disabled in this condition")
        os_path, rel = self._resolve(path, must_exist=False)
        if not rel:
            raise ToolError(f"{path}: Is a directory")
        if os_path.is_dir():
            raise ToolError(f"{path}: Is a directory")
        os_path.parent.mkdir(parents=True, exist_ok=True)
        os_path.write_text(content, encoding="utf-8")
        self._cur["written"].append(rel)
        self._on_write()
        return f"Wrote {len(content.encode('utf-8'))} bytes to {self._v(rel)}"

    def _on_write(self) -> None:
        pass


class ShellTools(ToolSet):
    name = "shell"
    tool_names = ("ls", "cd", "pwd", "find", "grep", "cat", "convert_to_text")

    def _t_ls(self, path: str = ".", long: bool = False) -> str:
        return self._list(path, long)

    def _t_cd(self, path: str) -> str:
        _, rel = self._resolve(path, kind="dir")
        self.cwd = self._v(rel)
        return self.cwd

    def _t_pwd(self) -> str:
        return self.cwd

    def _t_find(self, path: str = ".", name: str | None = None, type: str | None = None,
                maxdepth: int | None = None) -> str:
        from fnmatch import fnmatchcase

        os_path, rel = self._resolve(path, kind="dir")
        lines = []
        for child_rel, is_dir, _ in self._walk(os_path, rel, maxdepth):
            if type == "f" and is_dir or type == "d" and not is_dir:
                continue
            if name and not fnmatchcase(posixpath.basename(child_rel), name):
                continue
            if not is_dir:
                self._cur["seen"].append(child_rel)
            lines.append(self._v(child_rel))
        return "\n".join(sorted(lines)) if lines else "(no matches)"

    def _t_grep(self, pattern: str, path: str = ".", ignore_case: bool = False, files_only: bool = False) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}")
        os_path, rel = self._resolve(path)
        targets = [rel] if os_path.is_file() else self._all_files(os_path, rel)
        out = []
        for t in targets:
            data = self.root.joinpath(*t.split("/")).read_bytes()
            if is_binary(data):
                if rx.search(data.decode("latin-1")):
                    self._cur["seen"].append(t)
                    out.append(f"Binary file {self._v(t)} matches")
                continue
            hits = [(i, line) for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1)
                    if rx.search(line)]
            if hits:
                self._cur["seen"].append(t)
                if files_only:
                    out.append(self._v(t))
                else:
                    out.extend(f"{self._v(t)}:{i}:{line}" for i, line in hits)
        return "\n".join(out) if out else "(no matches)"

    def _t_cat(self, path: str) -> str:
        os_path, rel = self._resolve(path, kind="file")
        data = os_path.read_bytes()
        if is_binary(data):
            raise ToolError(f"{path}: binary file ({len(data)} bytes); use convert_to_text to extract its text")
        self._cur["read"].append(rel)
        return data.decode("utf-8", "replace")

    def _t_convert_to_text(self, path: str) -> str:
        _, rel = self._resolve(path, kind="file")
        text = self._text(rel)
        self._cur["read"].append(rel)
        return text


class FileTools(ToolSet):
    name = "files"
    tool_names = ("list_directory", "read_file", "glob", "search_files")

    def _t_list_directory(self, path: str = VROOT) -> str:
        return self._list(path)

    def _t_read_file(self, path: str) -> str:
        _, rel = self._resolve(path, kind="file")
        text = self._text(rel)
        self._cur["read"].append(rel)
        return text

    def _t_glob(self, pattern: str, path: str = VROOT) -> str:
        os_path, rel = self._resolve(path, kind="dir")
        rx = _glob_regex(pattern.lstrip("/"))
        base = len(rel) + 1 if rel else 0
        hits = [r for r in self._all_files(os_path, rel) if rx.fullmatch(r[base:])]
        self._cur["seen"].extend(hits)
        return "\n".join(self._v(h) for h in hits) if hits else "(no matches)"

    def _t_search_files(self, query: str, path: str = VROOT, max_results: int = 50) -> str:
        if not query.strip():
            raise ToolError("query must not be empty")
        os_path, rel = self._resolve(path, kind="dir")
        q = query.casefold()
        out = []
        for r in self._all_files(os_path, rel):
            snippet = None
            if q in posixpath.basename(r).casefold():
                snippet = "(file name match)"
            try:
                for line in self._text(r).splitlines():
                    if q in line.casefold():
                        snippet = line.strip()[:200]
                        break
            except ToolError:
                pass
            if snippet:
                self._cur["seen"].append(r)
                out.append(f"{self._v(r)}\n    {snippet}")
                if len(out) >= max_results:
                    out.append(f"(stopped after {max_results} results)")
                    break
        return "\n".join(out) if out else "(no matches)"


class IndexedTools(FileTools):
    name = "indexed"
    tool_names = FileTools.tool_names + ("search_index",)
    K1, B = 1.5, 0.75

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index: dict | None = None

    def _on_write(self) -> None:
        self._index = None

    def _build_index(self) -> dict:
        docs, df = {}, Counter()
        for r in self._all_files(self.root, ""):
            try:
                text = self._text(r)
            except ToolError:
                text = ""
            tf = Counter(_tokens(r.replace("/", " ")) + _tokens(text))
            docs[r] = (tf, sum(tf.values()), text)
            df.update(tf.keys())
        avg = sum(n for _, n, _ in docs.values()) / max(len(docs), 1)
        return {"docs": docs, "df": df, "avg": avg, "n": len(docs)}

    def _t_search_index(self, query: str, k: int = 10) -> str:
        if self._index is None:
            t0 = time.perf_counter()
            self._index = self._build_index()
            self._cur.setdefault("index_build_ms", [0.0])[0] = (time.perf_counter() - t0) * 1000
        idx = self._index
        terms = list(dict.fromkeys(_tokens(query)))
        if not terms:
            raise ToolError("query has no searchable terms")
        scores = []
        for r, (tf, length, _) in idx["docs"].items():
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1 + (idx["n"] - idx["df"][t] + 0.5) / (idx["df"][t] + 0.5))
                    s += idf * f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * length / idx["avg"]))
            if s > 0:
                scores.append((s, r))
        scores.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for rank, (s, r) in enumerate(scores[:k], 1):
            self._cur["seen"].append(r)
            text = idx["docs"][r][2]
            best = max(text.splitlines() or [""], key=lambda line: len(set(_tokens(line)) & set(terms)))
            out.append(f"{rank}. {self._v(r)}  (score {s:.2f})\n    {best.strip()[:200]}")
        return "\n".join(out) if out else "(no matches)"


TOOLSETS: dict[str, type[ToolSet]] = {"shell": ShellTools, "files": FileTools, "indexed": IndexedTools}


def make_toolset(name: str, root: str | Path, tracer: Tracer | None = None, **kwargs) -> ToolSet:
    if name not in TOOLSETS:
        raise ValueError(f"unknown toolset {name!r}; choose from {sorted(TOOLSETS)}")
    return TOOLSETS[name](root, tracer, **kwargs)


def render_context(root: str | Path, *, include_paths: bool = True) -> str:
    """The no-filesystem condition: every file's text concatenated into one prompt block."""
    ts = FileTools(root)
    parts = []
    for r in ts._all_files(ts.root, ""):
        try:
            text = ts._text(r)
        except ToolError:
            continue
        header = f"=== {VROOT}/{r} ===" if include_paths else "=== document ==="
        parts.append(f"{header}\n{text.rstrip()}\n")
    return "\n".join(parts)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _glob_regex(pattern: str) -> re.Pattern:
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _unique(xs: list) -> list:
    return list(dict.fromkeys(xs))
