import json
from copy import deepcopy
from pathlib import Path

import pytest

from fsbench.benchmark import load_manifest
from fsbench.calibration import calibrate
from fsbench.cli import main
from fsbench.evaluate import evaluate_runs, find_runs, keep_paired
from fsbench.human import human_run
from fsbench.models import register_provider
from fsbench.realistic import generate_realistic
from fsbench.taskpacks import build_pack, check_version
from fsbench.validity import leakage_findings, validate_benchmark


def test_pack_world_split_isolation_version_and_private_seeds(tmp_path):
    seeds = tmp_path/"seeds.json"
    seeds.write_text(json.dumps([10**30+i for i in range(10)]))
    root = tmp_path/"pack"
    data = build_pack(root,num_worlds=10,seed_file=seeds)
    assert data["tasks"]==50 and data["worlds"]==10
    sets = {split:{i["world_id"] for i in data["instances"] if i["split"]==split} for split in ("dev","validation","test")}
    assert not sets["dev"]&sets["test"] and not sets["validation"]&sets["test"]
    for row in data["instances"]:
        env = root/row["path"]
        if row["split"]!="dev":
            assert not (env/"manifest.json").exists()
            assert load_manifest(env)["split"]==row["split"]
    public_index = (root/"benchmark.json").read_text()
    assert str(10**30) not in public_index
    with pytest.raises(ValueError,match="immutable"):
        build_pack(root,num_worlds=10,seed_file=seeds)


def test_recipe_change_requires_new_benchmark_version(monkeypatch):
    monkeypatch.setattr("fsbench.taskpacks.recipe_digest",lambda:"changed")
    with pytest.raises(ValueError,match="new benchmark version"):
        check_version("realistic-v1")


def test_structural_leakage_scan_distinguishes_answers_from_labels(tmp_path):
    (tmp_path/"invoice.txt").write_text("Invoice amount: 900. This is legitimate source data.")
    assert leakage_findings(tmp_path)==[]
    (tmp_path/"required_document_1.json").write_text('{"is_answer":true,"ground_truth":900}')
    kinds = {r["kind"] for r in leakage_findings(tmp_path)}
    assert kinds=={"filename_metadata_leak","content_metadata_leak"}


def test_validator_uses_content_oracle_not_ground_truth_echo(tmp_path):
    root = tmp_path/"pack"
    envs = generate_realistic("research",32,root,level="light")
    report = validate_benchmark(root)
    assert report["automated_valid"] and report["oracle_success"]==5
    path = envs[0]/"manifest.json"
    m = load_manifest(envs[0])
    m["ground_truth"]["answer"] += 1
    path.write_text(json.dumps(m))
    report = validate_benchmark(root)
    assert not report["automated_valid"] and report["oracle_success"]==4


def test_human_collection_fixture_is_instrumented_and_excludes_ratings_from_time(tmp_path):
    env = generate_realistic("legal",11,tmp_path/"envs",level="light")[3]
    m = load_manifest(env)
    req = next(f for f in m["files"] if f["doc_id"]=="signed")
    responses = iter(["yes",'read_file '+json.dumps({"path":"/workspace/"+req["path"]}),"finish",
                      json.dumps({"answer":m["ground_truth"]["answer"]}),json.dumps([req["path"]]),"3","1,3"])
    output = tmp_path/"human"
    result = human_run(env,"anonymous_fixture",output,input_fn=lambda p:next(responses),output_fn=lambda p:None)
    assert result["success"] and result["files_opened"]==1 and result["searches"]==0
    assert result["difficulty_rating"]==3 and result["difficulty_causes"]==["finding_files","determining_latest_version"]
    assert result["evidence_recall"]==.5
    assert "tokens" not in json.dumps(result)
    assert result["measurement_mode"]=="instrumented_cli"
    assert (output/"human.json").exists()


def test_human_decline_and_abort_do_not_create_baselines(tmp_path):
    env = generate_realistic("hr",5,tmp_path/"envs",level="light")[0]
    out = tmp_path/"declined"
    assert human_run(env,"anonymous_fixture",out,input_fn=lambda p:"no",output_fn=lambda p:None) is None
    assert not out.exists()
    replies = iter(["yes","abort"])
    out = tmp_path/"aborted"
    assert human_run(env,"anonymous_fixture",out,input_fn=lambda p:next(replies),output_fn=lambda p:None) is None
    assert not (out/"human.json").exists() and not find_runs(out)


def test_calibration_requires_independent_humans_and_preserves_uncertainty(tmp_path):
    root = tmp_path/"human"
    root.mkdir()
    for i in range(3):
        folder = root/str(i)
        folder.mkdir()
        (folder/"human.json").write_text(json.dumps({"benchmark_version":"realistic-v1","env_id":"instance", "measurement_mode":"instrumented_cli",
            "participant":"anonymous_one","wall_time_s":100+i,"success":True,"difficulty_rating":5}))
    report = calibrate(None,root,tmp_path/"calibration")
    assert report["tasks"][0]["human_participants"]==1
    assert report["tasks"][0]["observed_difficulty"]=="insufficient human observations"
    assert report["tasks"][0]["agent_success_rate"] is None


def test_realistic_runner_pairs_models_on_actual_world_conditions(tmp_path):
    root = tmp_path/"envs"
    for seed,level in ((1,"light"),(2,"dense")):
        generate_realistic("legal",seed,root,level=level,templates=["legal.renewal"])
    class Fake:
        def __init__(self,model,**kwargs):
            self.name,self.model="fixture:"+model,model
        def describe(self):
            return {"model":self.model}
        def run(self,prompt,tools):
            return {}
    register_provider("realistic_fixture",Fake)
    out = tmp_path/"runs"
    main(["run","--envs",str(root),"--model","a","--model","b","--provider","realistic_fixture","--out",str(out)])
    rows = evaluate_runs(out,paired=True)
    assert len(rows)==4
    assert len(keep_paired(rows[:-1]))==2
    assert {r["benchmark_version"] for r in rows}=={"realistic-v1"}


def test_native_baseline_unknown_history_is_not_zero(tmp_path,monkeypatch):
    import os
    if hasattr(os,"startfile"):
        monkeypatch.setattr(os,"startfile",lambda path:None)
    env = generate_realistic("legal",8,tmp_path/"envs",level="light")[1]
    m = load_manifest(env)
    responses = iter(["yes",json.dumps({"answer":m["ground_truth"]["answer"]}),"","","","",""])
    out = tmp_path/"human"
    result = human_run(env,"anonymous_fixture",out,interface="native",input_fn=lambda p:next(responses),output_fn=lambda p:None)
    assert result["files_opened"] is None and result["searches"] is None
    metrics = json.loads((out/"metrics.json").read_text())
    assert metrics["files_read"] is None and metrics["evidence_found"] is None
    assert metrics["success"]


def test_new_testset_rejects_reused_worlds_before_writing(tmp_path):
    root = tmp_path/"base"
    seeds = tmp_path/"seeds.json"
    seeds.write_text("[123]")
    build_pack(root,num_worlds=1,seed_file=seeds)
    with pytest.raises(ValueError,match="overlap"):
        build_pack(tmp_path/"test",num_worlds=1,seed_file=seeds,test_only=True,exclude_root=root)
    assert not (tmp_path/"test").exists()


def test_human_answer_timer_excludes_post_answer_questions(tmp_path):
    import time
    env = generate_realistic("legal",4,tmp_path/"envs",level="light")[1]
    m = load_manifest(env)
    def reply(prompt):
        if prompt.startswith("Voluntary"):
            return "yes"
        if prompt == "> ":
            return "finish"
        if prompt.startswith("Final answer"):
            return json.dumps({"answer":m["ground_truth"]["answer"]})
        if prompt.startswith("Supporting"):
            time.sleep(.04)
        return ""
    out = tmp_path/"human"
    human_run(env,"anonymous_fixture",out,input_fn=reply,output_fn=lambda p:None)
    meta = json.loads((out/"meta.json").read_text())
    assert meta["session_wall_time_s"]-meta["wall_time_s"]>=.03
