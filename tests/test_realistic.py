import json
from copy import deepcopy
from pathlib import Path

import pytest

from fsbench.benchmark import load_manifest
from fsbench.citations import score_citations
from fsbench.realistic import DOMAINS, generate_realistic, world_material, materialize_world, score_realistic
from fsbench.runner import OracleAgent, run_agent
from fsbench.experiment import build_schedule, validate_environment
from fsbench.tools import make_toolset


@pytest.mark.parametrize("domain",DOMAINS)
def test_five_distinct_templates_solved_from_visible_contents(domain,tmp_path):
    envs = generate_realistic(domain,17,tmp_path/domain,level="routine")
    assert len(envs)==5
    for env in envs:
        m = load_manifest(env)
        assert m["benchmark_family"]=="realistic" and m["benchmark_version"]=="realistic-v1"
        result = run_agent(env,OracleAgent(m),tmp_path/"runs"/m["env_id"])
        assert result["success"], (m["task_template"],result["outcome"],result.get("agent_error"))
        assert result["evidence_precision"]==result["evidence_recall"]==1
        assert result["answer_correct"]
        validate_environment(env,check_index=True)


def test_same_world_shared_bytes_and_difficulty_invariance(tmp_path):
    variants = [materialize_world("finance",19,level) for level in ("light","routine","dense")]
    assert variants[0][2]==variants[1][2]==variants[2][2]
    for records,blobs,_ in variants[1:]:
        for r in variants[0][0]:
            if r["doc_id"]!="file-activity":
                assert blobs[r["path"]]==variants[0][1][r["path"]]
    envs = generate_realistic("finance",19,tmp_path/"envs")
    inventories = [{p.relative_to(e/"workspace").as_posix():p.read_bytes() for p in (e/"workspace").rglob("*") if p.is_file()} for e in envs]
    assert all(i==inventories[0] for i in inventories)


def test_citations_tolerate_formatting_but_not_wrong_sources():
    m = {"citation_required":True,"required_doc_ids":["a","b"],"files":[
        {"path":"finance/Invoice.pdf","doc_id":"a"},{"path":"pay/ledger.csv","doc_id":"b"},
        {"path":"old/ledger.csv","doc_id":"old"}]}
    score = score_citations({"evidence":["[invoice](/workspace/finance/Invoice.pdf#L2)","pay\\ledger.csv:12"]},m)
    assert score["evidence_recall"]==score["evidence_precision"]==1
    score = score_citations({"sources":["Invoice.pdf","ledger.csv","../manifest.json"]},m)
    assert score["evidence_recall"]==.5 and score["evidence_precision"]==pytest.approx(1/3)
    assert len(score["invalid_citations"])==2


def test_bad_citation_does_not_fail_correct_answer(tmp_path):
    env = generate_realistic("legal",7,tmp_path/"envs")[0]
    m = load_manifest(env)
    class Uncited:
        name="uncited"
        def run(self,prompt,tools):
            return {"answer":m["ground_truth"]["answer"],"evidence":["missing.pdf"]}
    result = run_agent(env,Uncited(),tmp_path/"run")
    assert result["success"] and result["answer_correct"]
    assert result["evidence_recall"]==result["evidence_precision"]==0


@pytest.mark.parametrize("fields,expected", [
    ({"file_exists": False}, "write_failure"),
    ({"file_exists": True, "header": False}, "output_format_failure"),
])
def test_realistic_workflow_output_failure_classification(tmp_path, fields, expected):
    from fsbench.evaluate import classify_failure
    manifest = {"task": {"type": "finance.payment_summary",
                         "output_path": "output/summary.csv",
                         "answer_schema": {"answer": "string", "evidence": "array"}}}
    result = classify_failure({"success": False, "fields": fields},
                              {"trap_exposed": False}, {}, {}, manifest, tmp_path)
    assert result["failure_type"] == expected


def test_private_seeds_and_labels_not_exposed(tmp_path):
    private = tmp_path/".private"
    envs = generate_realistic("engineering",923487129487129487,tmp_path/"test",split="test",evaluator_root=private)
    for env in envs:
        assert not (env/"manifest.json").exists()
        public = (env/"task.json").read_text()+(env/"evaluation_ref.json").read_text()
        assert "923487129487129487" not in public
        m = load_manifest(env)
        for ts in ("shell","files","indexed"):
            tools = make_toolset(ts,env/"workspace")
            name = "cat" if ts=="shell" else "read_file"
            assert tools.call(name,{"path":"/workspace/../manifest.json"}).startswith("Error:")
        run = tmp_path/"runs"/m["env_id"]
        run_agent(env,OracleAgent(m),run)
        assert "923487129487129487" not in (run/"meta.json").read_text()
        from fsbench.cli import main
        with pytest.raises(ValueError,match="evaluator"):
            main(["inspect","--env",str(env)])


def test_metadata_history_and_byte_sameness(tmp_path):
    env = generate_realistic("legal",1,tmp_path/"envs")[0]
    m = load_manifest(env)
    tools = make_toolset("files",env/"workspace",include_timestamps=True)
    f = next(f for f in m["files"] if f["role"]=="trap")
    actual = json.loads(tools.call("stat_file",{"path":"/workspace/"+f["path"]}))
    assert actual["modified_at"]==f["modified_at"]
    schedule = build_schedule([env],["files","indexed","shell"])
    assert len({c["workspace_sha256"] for c in schedule})==1


def test_boolean_tasks_do_not_have_constant_answers():
    finance = [world_material("finance",s)[1][-1]["answer"] for s in range(15)]
    research = [world_material("research",s)[1][-1]["answer"] for s in range(15)]
    assert set(finance)==set(research)=={True,False}
