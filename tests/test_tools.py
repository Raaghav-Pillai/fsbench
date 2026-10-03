import pytest

from fsbench import FSConfig
from fsbench.tools import FileTools, IndexedTools, ShellTools, Tracer, make_toolset


@pytest.fixture
def pdf_env(make_env):
    env, m = make_env("conflict", FSConfig(mime_types=("pdf",), distractors=15))
    return env / "workspace", m


def test_shell_grep_is_blind_to_pdf_but_search_files_is_not(pdf_env):
    ws, m = pdf_env
    contract_id = m["ground_truth"]["contract_id"]
    required = next(f["path"] for f in m["files"] if f["role"] == "required")
    shell = ShellTools(ws)
    assert shell.call("grep", {"pattern": contract_id, "path": "/workspace"}) == "(no matches)"
    assert "binary file" in shell.call("cat", {"path": f"/workspace/{required}"})
    assert contract_id in shell.call("convert_to_text", {"path": f"/workspace/{required}"})

    files = FileTools(ws)
    assert required in files.call("search_files", {"query": contract_id})


def test_index_ranks_the_relevant_file_highly(pdf_env):
    ws, m = pdf_env
    client = m["task"]["prompt"].split("in force with ")[1].split("?")[0]
    idx = IndexedTools(ws)
    out = idx.call("search_index", {"query": f"executed master services agreement {client}", "k": 10})
    required = next(f["path"] for f in m["files"] if f["role"] == "required")
    assert required in out


def test_paths_are_virtual_case_sensitive_and_confined(make_env):
    env, m = make_env("retrieve", FSConfig(mime_types=("txt",)))
    tracer = Tracer()
    t = ShellTools(env / "workspace", tracer)
    real = m["files"][0]["path"]
    top = real.split("/")[0]

    assert "Error" not in t.call("ls", {"path": top})
    assert "No such file" in t.call("ls", {"path": top.upper()})
    assert "only /workspace" in t.call("cat", {"path": "/etc/passwd"})
    assert "only /workspace" in t.call("ls", {"path": "/workspace/../.."})
    t.call("cd", {"path": top})
    assert t.call("pwd") == f"/workspace/{top}"
    assert "Employee Record" in t.call("cat", {"path": "/workspace/" + real})

    ev = tracer.events
    assert [e["tool"] for e in ev] == ["ls", "ls", "cat", "ls", "cd", "pwd", "cat"]
    assert ev[1]["missing"] == [f"/workspace/{top.upper()}"]
    assert ev[2]["missing"] == ["/etc/passwd"]
    assert ev[6]["read"] == [real]
    assert ev[5]["path_args"] == 0


def test_glob_find_and_listing_record_seen_files(make_env):
    env, m = make_env("reconcile", FSConfig(mime_types=("txt", "csv")))
    tracer = Tracer()
    t = FileTools(env / "workspace", tracer)
    out = t.call("glob", {"pattern": "**/*.csv"})
    csvs = [f["path"] for f in m["files"] if f["format"] == "csv"]
    assert csvs and all(p in out for p in csvs)
    assert set(tracer.events[-1]["seen"]) == set(csvs)

    s = ShellTools(env / "workspace", tracer)
    found = s.call("find", {"path": "/workspace", "name": "*.txt", "type": "f"})
    assert all(f["path"] in found for f in m["files"] if f["format"] == "txt")


def test_writes_can_be_disabled_and_bad_args_are_reported(make_env):
    env, _ = make_env("retrieve")
    ro = make_toolset("files", env / "workspace", allow_writes=False)
    assert "write_file" not in {s["name"] for s in ro.specs()}
    assert "unknown tool" in ro.call("write_file", {"path": "x.txt", "content": "hi"})
    assert "bad arguments" in ro.call("read_file", {"file": "x"})

    rw = make_toolset("files", env / "workspace")
    assert "Wrote" in rw.call("write_file", {"path": "/workspace/notes/scratch.md", "content": "memo"})
    assert rw.call("read_file", {"path": "/workspace/notes/scratch.md"}) == "memo"
    assert rw.tracer.events[0]["written"] == ["notes/scratch.md"]


def test_output_truncation_is_traced(make_env):
    env, _ = make_env("reconcile", FSConfig(distractors=60))
    t = ShellTools(env / "workspace", max_output_chars=300)
    out = t.call("find", {"path": "/workspace"})
    assert "output truncated" in out
    assert t.tracer.events[-1]["truncated"]


def test_spec_styles():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for name in ("shell", "files", "indexed"):
            ts = make_toolset(name, d)
            a, o = ts.specs("anthropic"), ts.specs("openai")
            assert len(a) == len(o) == len(ts.available)
            assert all("input_schema" in s for s in a)
            assert all(s["type"] == "function" for s in o)
