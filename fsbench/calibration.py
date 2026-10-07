"""Descriptive observed difficulty, stratified by model/interface; never a universal score."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from fsbench.evaluate import evaluate_runs
from fsbench.statistics import write_csv


def calibrate(runs, human_runs, out, *, min_humans=3):
    if min_humans<2:
        raise ValueError("at least two independent participants are required for calibration")
    rows = evaluate_runs(runs) if runs and Path(runs).exists() else []
    humans = [json.loads(p.read_text(encoding="utf-8")) for p in Path(human_runs).rglob("human.json")] if human_runs and Path(human_runs).exists() else []
    human_groups = defaultdict(list)
    for h in humans:
        human_groups[(h["benchmark_version"],h["env_id"],h["measurement_mode"])].append(h)
    agent_groups = defaultdict(list)
    for r in rows:
        if not r["agent"].startswith("human:"):
            agent_groups[(r["benchmark_version"],r["env_id"],r.get("model"),r["toolset"],r["tool_policy"])].append(r)
    result = []
    all_tasks = {(k[0],k[1]) for k in human_groups} | {(k[0],k[1]) for k in agent_groups}
    for version,instance in sorted(all_tasks):
        hkeys = [k for k in human_groups if k[:2]==(version,instance)] or [(version,instance,"none")]
        akeys = [k for k in agent_groups if k[:2]==(version,instance)] or [(version,instance,None,None,None)]
        for hk in hkeys:
            hs = human_groups[hk]
            # One participant contributes one value per task, even after repeated attempts.
            per_person = defaultdict(list)
            for h in hs:
                per_person[h["participant"]].append(h)
            times = [median(h["wall_time_s"] for h in attempts) for attempts in per_person.values()]
            ratings = [median(h["difficulty_rating"] for h in attempts if h.get("difficulty_rating"))
                       for attempts in per_person.values() if any(h.get("difficulty_rating") for h in attempts)]
            perceived = median(ratings) if ratings else None
            label = "insufficient human observations"
            if len(ratings)>=min_humans:
                label = "low observed difficulty" if perceived<=2 else "high observed difficulty" if perceived>=4 else "medium observed difficulty"
            for ak in akeys:
                agents = agent_groups[ak]
                result.append({"benchmark_version":version,"env_id":instance,"model":ak[2],"toolset":ak[3],"tool_policy":ak[4],
                    "human_measurement_mode":hk[2],"human_participants":len(per_person),"human_sessions":len(hs),
                    "median_human_completion_time_s":median(times) if times else None,
                    "human_success_rate":sum(sum(h["success"] for h in attempts)/len(attempts) for attempts in per_person.values())/len(per_person) if per_person else None,
                    "median_human_rating":perceived,"observed_difficulty":label,
                    "agent_runs":len(agents),"agent_success_rate":sum(r["success"] for r in agents)/len(agents) if agents else None,
                    "median_agent_calls":median(r["n_calls"] for r in agents) if agents else None,
                    "retrieval_failure_rate":sum(r.get("task_failure_type") in ("retrieval_failure","path_hallucination") for r in agents)/len(agents) if agents else None})
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"calibration.csv",result)
    payload = {"status":"descriptive calibration; not external-validity proof", "minimum_independent_human_ratings":min_humans,
        "classification_rule":"Median participant difficulty rating <=2 low, >=4 high, otherwise medium; insufficient ratings remain unclassified. No filesystem-variable sum is used.",
        "limitations":"No causal claim or human-agent correlation is inferred automatically; sibling tasks and repeat participants require clustered study analysis. Native timing histories are self-reported.","tasks":result}
    (out/"calibration.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload
