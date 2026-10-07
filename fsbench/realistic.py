"""realistic-v1 candidate: authored fictional business tasks, not collected personal data.

Each world supplies five tasks over identical file bytes. Level changes add clutter,
copies and misleading timestamps; authoritative facts and required identities stay fixed.
"""
import csv
import hashlib
import io
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fsbench.config import FSConfig
from fsbench.docs import Doc, Table
from fsbench.render import render
from fsbench.rng import keyed_rng

VERSION = "realistic-v1"
DOMAINS = ("finance", "legal", "hr", "research", "engineering")
LEVELS = ("light", "routine", "dense")


def world_material(domain, seed):
    rng = keyed_rng(VERSION, domain, seed)
    code = rng.randint(10000, 99999)
    company = f"Juniper Studio {code}"
    docs, tasks = [], []

    def doc(key, path, title, *, fields=(), columns=(), rows=(), paragraphs=(), role="generic", date="2026-09-30"):
        d = Doc(key, domain, title, Path(path).stem, tuple(Path(path).parts[:-1]),
                list(fields), Table(list(columns), [[str(c) for c in row] for row in rows]) if columns else None,
                list(paragraphs))
        docs.append({"doc": d, "path": path, "role": role, "date": date})

    def task(name, caps, question, answer, evidence, *, output=None, rows=None):
        tasks.append({"template": f"{domain}.{name}", "capabilities": caps, "question": question,
                      "answer": answer, "required": evidence, "output": output, "rows": rows})

    if domain == "finance":
        amounts = [rng.randrange(8, 40)*100, rng.randrange(10, 50)*100, rng.randrange(5, 20)*100]
        paid = [amounts[0], amounts[1]//2, 0]
        ids = [f"INV-{code+i}" for i in range(3)]
        credit = 100
        if rng.choice([False, True]):
            paid = [amounts[0], amounts[1], amounts[2]-credit]
        balance = sum(amounts)-sum(paid)-credit
        doc("invoices", "company/invoices/Invoice register.xlsx", "Posted invoice register", fields=[("Client", company)],
            columns=["invoice_id", "issued", "amount", "status"],
            rows=[[ids[i], f"2026-09-{10+i*5}", amounts[i], "posted"] for i in range(3)])
        doc("payments", "company/accounting/export (3).csv", "Bank receipts", columns=["invoice_id", "received", "amount"],
            rows=[[ids[i], "2026-09-28", paid[i]] for i in range(3) if paid[i]])
        doc("credit", "company/clients/Payment adjustment FINAL.docx", "Executed payment adjustment",
            fields=[("Client", company), ("Approved credit", str(credit)), ("Applies to", ids[2]), ("Status", "approved")])
        doc("terms", "company/clients/Service agreement.pdf", "Signed payment terms",
            fields=[("Client", company), ("Payment deadline", "2026-09-30"), ("Required settlement", "all posted invoices less approved credits")])
        doc("old", "company/archive/Invoice register FINAL.xlsx", "Superseded invoice register", role="trap",
            columns=["invoice_id", "issued", "amount", "status"], rows=[[ids[-1]+"-DRAFT", "2026-10-01", 9999, "draft"]])
        summary = [[ids[i], amounts[i]-paid[i]-(credit if i==2 else 0)] for i in range(3) if amounts[i]-paid[i]-(credit if i==2 else 0)>0]
        task("unpaid", ["reconciliation", "multi_file_reasoning"], f"For {company}, which posted invoices remain unpaid as of September 30, 2026? Return their invoice IDs.",
             [r[0] for r in summary], ["invoices", "payments", "credit"])
        task("balance", ["reconciliation", "multi_file_reasoning", "provenance"], f"Calculate {company}'s outstanding balance after receipts and the approved credit as of September 30, 2026. Trace the invoice, receipt, and approved-credit sources.",
             balance, ["invoices", "payments", "credit"])
        task("latest_invoice", ["temporal_reasoning", "source_selection"], f"Identify {company}'s most recently issued posted invoice as of September 30, 2026. Exclude drafts; use issue dates, not file modification times.",
             ids[-1], ["invoices"])
        task("payment_summary", ["workflow_execution", "reconciliation"], f"Create output/payment_summary.csv for {company}'s unpaid posted invoices, with columns invoice_id,amount_due. Apply the approved credit.",
             "output/payment_summary.csv", ["invoices", "payments", "credit"], output="output/payment_summary.csv", rows=summary)
        task("compliance", ["source_selection", "reconciliation", "multi_file_reasoning"], f"As of September 30, 2026, has {company} fulfilled the signed agreement's payment terms? Return a boolean after checking invoices, receipts and the approved adjustment.",
             balance<=0, ["terms", "invoices", "payments", "credit"])
    elif domain == "legal":
        agreement = f"MSA-{code}"
        old_value, increase = rng.randrange(50, 150)*1000, rng.randrange(5, 20)*1000
        value = old_value+increase
        renewal = f"2027-{rng.randint(1,9):02d}-15"
        doc("signed", "legal/signed/IMG_3812.pdf", "Executed master services agreement",
            fields=[("Client", company), ("Agreement", agreement), ("Status", "signed by both parties"),
                    ("Effective date", "2026-01-01"), ("Annual value", str(old_value)), ("Renewal date", "2027-01-01")])
        doc("amendment", "legal/client_uploads/Meeting Notes 6-12.docx", "Executed amendment 1",
            fields=[("Agreement", agreement), ("Effective date", "2026-06-12"), ("Status", "signed by both parties"),
                    ("Annual value", str(value)), ("Renewal date", renewal)], paragraphs=["These terms replace the corresponding terms of the executed agreement."])
        doc("register", "legal/contracts/Register.csv", "Agreement execution register",
            columns=["agreement", "state", "amendment", "effective"], rows=[[agreement, "in force", "1 signed", "2026-06-12"], [agreement+"-B", "unsigned draft", "none", "not effective"]])
        doc("prior", "legal/old/Service agreement 2025.pdf", "Expired annual agreement",
            fields=[("Agreement", agreement+"-2025"), ("Annual value", str(old_value-1000)), ("Expiry", "2025-12-31")], role="trap")
        doc("draft", "legal/drafts/contract_FINAL_v2.pdf", "Unsigned proposed replacement",
            fields=[("Agreement", agreement+"-B"), ("Annual value", str(value+9000)), ("Status", "draft; neither party has signed")], role="trap", date="2026-10-02")
        task("current_terms", ["source_selection", "multi_file_reasoning"], f"Report the agreement ID and annual value currently in force for {company} on September 30, 2026, incorporating any executed amendment.",
             {"agreement": agreement, "annual_value": value}, ["signed", "amendment", "register"])
        task("renewal", ["retrieval", "temporal_reasoning"], f"What is the effective renewal date for {company}'s signed agreement as of September 30, 2026, including amendments?", renewal, ["signed", "amendment"])
        task("compare_terms", ["multi_file_reasoning", "temporal_reasoning"], f"How much higher is {company}'s current annual contract value than the expired 2025 agreement?", value-(old_value-1000), ["signed", "amendment", "prior"])
        task("authority", ["source_selection"], f"Which agreement ID is authoritative for {company} on September 30, 2026? Use the execution register and signed document, not the newest filename.", agreement, ["signed", "register"])
        task("terms_provenance", ["provenance", "multi_file_reasoning"], f"Trace {company}'s effective annual value from execution status through the base agreement and amendment. Return the value and cite all three links in this evidence chain.", value, ["register", "signed", "amendment"])
    elif domain == "hr":
        person = f"Avery Vale {code}"
        manager = rng.choice(["Morgan Lee", "Taylor Quinn", "Jordan Ellis"])
        salary, bonus = rng.randrange(65, 100)*1000, rng.randrange(1, 5)*500
        rating = rng.choice(["meets expectations", "exceeds expectations"])
        doc("employee", "hr/employees/Employee record.pdf", "Active employee record",
            fields=[("Employee", person), ("Employee ID", f"E-{code}"), ("Manager", "Casey Rowan"), ("Annual salary", str(salary)), ("Status", "active")])
        doc("transfer", "hr/onboarding/Untitled document.docx", "Approved reporting-line change",
            fields=[("Employee", person), ("Effective", "2026-08-01"), ("Manager", manager), ("Approval", "HR operations approved")])
        doc("payroll", "hr/payroll/September payroll.xlsx", "September payroll calculation",
            fields=[("Employee", person)], columns=["component", "amount"], rows=[["base monthly", f"{salary/12:.2f}"], ["approved bonus", bonus]])
        doc("review", "hr/reviews/Review FINAL.pdf", "Approved employee review",
            fields=[("Employee", person), ("Review period end", "2026-09-30"), ("Approved", "yes"), ("Rating", rating)])
        doc("review_old", "hr/archive/Copy of Review.pdf", "Previous employee review", role="trap",
            fields=[("Employee", person), ("Review period end", "2025-09-30"), ("Approved", "yes"), ("Rating", "needs improvement")])
        doc("checklist", "hr/onboarding/Checklist.csv", "Onboarding status", columns=["employee", "item", "complete"],
            rows=[[person, "identity verification", "yes"], [person, "security training", "no"], [person, "benefits election", "no"]])
        task("manager", ["retrieval", "source_selection"], f"Who is {person}'s current manager as of September 30, 2026? Reconcile the employee record with approved changes.", manager, ["employee", "transfer"])
        task("payroll", ["reconciliation", "multi_file_reasoning"], f"Calculate {person}'s gross September pay: annual base salary divided by 12, rounded to cents, plus the approved payroll bonus.", round(salary/12,2)+bonus, ["employee", "payroll"])
        task("latest_review", ["temporal_reasoning", "source_selection"], f"What rating did {person} receive in the latest approved review by review-period end date as of September 30, 2026?", rating, ["review"])
        task("onboarding", ["workflow_execution"], f"Create output/onboarding.csv listing {person}'s incomplete onboarding items, with columns employee,item.", "output/onboarding.csv", ["checklist"],
             output="output/onboarding.csv", rows=[[person,"security training"], [person,"benefits election"]])
        task("manager_provenance", ["provenance", "temporal_reasoning"], f"Trace the evidence establishing {person}'s current manager: identify the baseline record and the approved effective change. Return the manager and cite both sources.", manager, ["employee", "transfer"])
    elif domain == "research":
        sales = [rng.randrange(6, 12)*100000 for _ in range(3)]
        refunds = [rng.randrange(1, 5)*10000 for _ in range(3)]
        complete = rng.choice([False,True])
        refund_rows = list(zip([7,8,9],refunds)) if complete else list(zip([7,8],refunds[:2]))
        net = sum(sales)-sum(r[1] for r in refund_rows)
        version = rng.randint(2,5)
        old_net = net-rng.randrange(1, 5)*10000
        study = f"Market study {code}"
        doc("sales", "research/datasets/sales_q3.csv", "Q3 posted sales", fields=[("Study", study)], columns=["month", "sales"], rows=list(zip([7,8,9],sales)))
        doc("refunds", "research/exports/export (3).xlsx", "Q3 refunds", fields=[("Study", study)], columns=["month", "refunds"], rows=refund_rows)
        doc("report", "research/notes/Q3 Budget FINAL.docx", "Issued Q3 research note", fields=[("Study", study), ("Version", str(version)), ("Issued", "2026-09-30"), ("Net revenue", str(net))],
            paragraphs=["Reported net revenue is the sum of exported sales rows for months 7, 8 and 9 less exported refunds for those months. Missing months are not imputed; coverage must be assessed separately. Sources: sales_q3.csv and export (3).xlsx."])
        doc("previous", "research/drafts/report_final_v2.pdf", "Superseded preliminary estimate", fields=[("Study", study), ("Version", str(version-1)), ("Net revenue", str(old_net)), ("Status", "superseded")], role="trap")
        doc("registry", "research/papers/Analysis index.csv", "Publication register", columns=["study", "version", "status"], rows=[[study,version-1,"superseded"],[study,version,"issued"]])
        doc("coverage", "research/notes/Collection notes.docx", "Required quarterly coverage", paragraphs=["The quarterly estimate must include months 7, 8 and 9 for both sales and refunds. The preliminary estimate is not an issued result."])
        task("claim_provenance", ["provenance", "multi_file_reasoning"], f"Reconstruct the issued Q3 net revenue for {study}. Return the number and cite the issued note plus both underlying source datasets, forming the complete evidence chain.", net, ["report", "sales", "refunds"])
        task("reconstruct", ["reconciliation", "multi_file_reasoning"], f"Independently calculate Q3 net revenue for {study} from the sales and refunds datasets, using months 7, 8 and 9.", net, ["sales", "refunds"])
        task("latest_analysis", ["source_selection", "temporal_reasoning"], f"Which analysis version is the issued result for {study}? Confirm against the publication register and the issued note; ignore filesystem timestamps.", version, ["registry", "report"])
        task("compare_estimate", ["multi_file_reasoning"], f"How much did {study}'s issued net revenue change relative to the superseded preliminary estimate? Return issued minus preliminary.", net-old_net, ["report", "previous"])
        task("coverage", ["reconciliation", "source_selection"], f"Do both source datasets for {study} contain an explicit row for every month required by the quarterly collection notes? Return a boolean and cite the notes and both datasets.", complete, ["coverage", "sales", "refunds"])
    elif domain == "engineering":
        base = rng.randint(80, 150)
        current = base+rng.randint(-20,30)
        batch = rng.choice([8,16,32,64])
        run = f"run-{code}"
        doc("results", "project/reports/Benchmark FINAL.csv", "Completed benchmark runs", columns=["run_id", "config", "completed", "median_ms", "status"],
            rows=[[run+"-A","baseline","2026-09-20",base,"complete"],[run+"-B","candidate","2026-09-29",current,"complete"]])
        doc("config", "project/data/config.json", "Candidate configuration", fields=[("Config ID", "candidate"), ("batch_size", str(batch)), ("precision", "fp16")])
        doc("log", "project/logs/run.log", "Execution log", paragraphs=[f"2026-09-29 completed {run}-B config=candidate; samples_ms={current-2},{current},{current+2}"])
        doc("release", "project/docs/Release notes.docx", "Accepted benchmark publication", fields=[("Accepted run", run+"-B"), ("Status", "approved for reporting")], paragraphs=["Use completed accepted runs. Filesystem copy times do not establish completion order."])
        doc("draft", "project/exports/Copy of Report.xlsx", "Aborted rerun", role="trap", columns=["run_id","median_ms","status"], rows=[[run+"-C",1,"aborted; incomplete samples"]])
        doc("threshold", "project/docs/Performance criteria.pdf", "Regression policy", paragraphs=["A latency regression occurs only if candidate median latency is more than 10 percent above baseline. Equality at 10 percent is not a regression."])
        task("latest_result", ["source_selection", "temporal_reasoning"], "Identify the latest completed benchmark run approved for reporting. Return its run ID.", run+"-B", ["results", "release"])
        task("config_provenance", ["provenance", "multi_file_reasoning"], "What batch size produced the accepted candidate result? Trace the accepted run through its execution log to the configuration and cite all three sources.", batch, ["release", "log", "config"])
        task("compare_runs", ["multi_file_reasoning", "reconciliation"], "Calculate candidate minus baseline median latency in milliseconds from completed runs, and verify the candidate median from raw log samples.", current-base, ["results", "log"])
        task("regression", ["source_selection", "multi_file_reasoning"], "Does the completed candidate run violate the documented latency regression policy relative to baseline? Return a boolean.", current>1.1*base, ["results", "threshold"])
        task("summary", ["workflow_execution", "provenance"], "Create output/benchmark_summary.csv for both completed runs with columns run_id,median_ms. Exclude aborted runs and verify the accepted candidate using release notes and log samples.",
             "output/benchmark_summary.csv", ["results", "release", "log"], output="output/benchmark_summary.csv", rows=[[run+"-A",base],[run+"-B",current]])
    else:
        raise ValueError(f"unknown workspace domain: {domain}")
    return docs, tasks


def materialize_world(domain, seed, level="routine"):
    if level not in LEVELS:
        raise ValueError(f"unknown difficulty level {level}")
    docs, tasks = world_material(domain, seed)
    records, blobs = [], {}
    for i, item in enumerate(docs):
        d, path = item["doc"], item["path"]
        fmt = "txt" if Path(path).suffix==".log" else Path(path).suffix[1:]
        data = render(d, fmt)
        # A stale file can have a recent copy time; authority comes from its contents.
        modified = "2026-10-03T09:00:00+00:00" if item["role"] == "trap" else f"{item['date']}T09:00:00+00:00"
        records.append({"path": path, "doc_id": d.doc_id, "role": item["role"], "kind": domain,
            "format": fmt, "copy": 0, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "document_sha256": hashlib.sha256(json.dumps(asdict(d), sort_keys=True).encode()).hexdigest(),
            "created_at": "2026-01-02T09:00:00+00:00", "modified_at": modified})
        blobs[path] = data
    rng = keyed_rng(VERSION, domain, seed, "clutter")
    count = {"light": 2, "routine": 12, "dense": 35}[level]
    root = {"finance":"company", "legal":"legal", "hr":"hr", "research":"research", "engineering":"project"}[domain]
    folders = {"finance":["downloads","tax","archive"], "legal":["downloads","old","drafts"],
        "hr":["archive","onboarding","reviews"], "research":["figures","drafts","exports"], "engineering":["misc","archive","exports"]}[domain]
    for i in range(count):
        path = f"{root}/{folders[i%3]}/" + [f"Invoice_{80000+i}.pdf", f"Meeting Notes 6-{i+1}.docx", f"export ({i+4}).csv"][i%3]
        d = Doc(f"memo-{i}", "memo", "Internal planning note", "note", (),
            fields=[("Project", f"Cedar Office {rng.randint(100,999)}"), ("Status", "planning only")],
            paragraphs=["Quarterly planning meeting: confirm the room booking and circulate the revised cost estimate before Friday. Purchase approval is pending."])
        data = render(d, Path(path).suffix[1:])
        records.append({"path":path,"doc_id":d.doc_id,"role":"generic","kind":"memo","format":Path(path).suffix[1:],
            "copy":0,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
            "document_sha256":hashlib.sha256(json.dumps(asdict(d),sort_keys=True).encode()).hexdigest(),
            "created_at":"2026-02-01T09:00:00+00:00","modified_at":"2026-10-01T09:00:00+00:00"})
        blobs[path] = data
    if level == "dense":
        for record in list(records[:2]):
            copy = {**record, "path": f"{root}/downloads/Copy of {Path(record['path']).name}", "copy":1}
            records.append(copy)
            blobs[copy["path"]] = blobs[record["path"]]
    history = io.StringIO()
    writer = csv.writer(history, lineterminator="\n")
    writer.writerow(["path","created_at","modified_at"])
    writer.writerows([r["path"],r["created_at"],r["modified_at"]] for r in records)
    path = f"{root}/File activity.csv"
    data = history.getvalue().encode()
    records.append({"path":path,"doc_id":"file-activity","role":"generic","kind":"history","format":"csv","copy":0,
        "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
        "created_at":"2026-10-04T09:00:00+00:00","modified_at":"2026-10-04T09:00:00+00:00"})
    blobs[path] = data
    return records, blobs, tasks


def generate_realistic(domain, seed, out, *, level="routine", split="dev", evaluator_root=None, templates=None):
    from fsbench.paths import long_path
    records, blobs, tasks = materialize_world(domain, seed, level)
    world_id = hashlib.sha256(f"{VERSION}\0{domain}\0{seed}".encode()).hexdigest()[:24]
    envs = []
    for task in tasks:
        if templates and task["template"] not in templates:
            continue
        identifier = hashlib.sha256(f"{world_id}\0{task['template']}\0{level}".encode()).hexdigest()[:32]
        env = long_path(Path(out) / identifier)
        if env.exists():
            raise ValueError(f"refusing to overwrite task instance: {env}")
        env.mkdir(parents=True)
        workspace = env / "workspace"
        for path, data in blobs.items():
            target = workspace / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for record in records:
            timestamp = datetime.fromisoformat(record["modified_at"]).timestamp()
            os.utime(workspace / record["path"], (timestamp,timestamp))
        schema = {"answer": "task answer (number, boolean, string, list or object as requested)", "evidence": ["workspace-relative source paths"]}
        citation_instruction = "Cite the source documents supporting your answer in evidence. Use workspace-relative paths."
        prompt = "You have access to a fictional workplace file share mounted at /workspace.\n\n" + task["question"] + "\n\n" + citation_instruction + "\nReply with a JSON object: " + json.dumps(schema)
        task_data = {"type":task["template"],"level":3,"task_id":identifier,"prompt":prompt,"answer_schema":schema,"output_path":task["output"]}
        files = [{**r, "role": "required" if r["doc_id"] in task["required"] else r["role"]} for r in records]
        cfg = FSConfig(distractors={"light":2,"routine":12,"dense":35}[level], mime_types=("pdf","xlsx","csv","docx","json","txt")).to_dict()
        gt = {"answer":task["answer"]}
        if task["output"]:
            gt.update(output_path=task["output"], rows=task["rows"],
                columns={"finance":"invoice_id,amount_due","hr":"employee,item","engineering":"run_id,median_ms"}[domain].split(","))
        manifest = {"manifest_version":3,"env_id":identifier,"benchmark_family":"realistic","benchmark_version":VERSION,
            "world_id":world_id,"workspace_domain":domain,"task_template":task["template"],"task_families":task["capabilities"],
            "difficulty_level":level,"split":split,"generation_seed":seed,
            "seeds":{"world":world_id,"task":task["template"],"layout":world_id},
            "config":cfg,"condition":{"label":level,"config_hash":hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:8]},
            "task":task_data,"ground_truth":gt,"citation_required":True,"required_doc_ids":task["required"],
            "trap_doc_ids":[r["doc_id"] for r in files if r["role"]=="trap"],"decoy_answers":{},"files":files,
            "oracle":{"min_reads":len(task["required"]),"min_writes":int(bool(task["output"])),
                "known_path_optimal_calls":len(task["required"])+int(bool(task["output"])),
                "discovery_optimal_calls":len(task["required"])+5+int(bool(task["output"]))},
            "stats":{"n_files":len(files),"n_dirs":len({str(Path(r['path']).parent) for r in files}),"max_depth":max(r['path'].count('/') for r in files)},
            "sweep":None,"filename_noise_version":2}
        (env / "task.json").write_text(json.dumps({k:task_data[k] for k in ("task_id","prompt","answer_schema")},indent=2),encoding="utf-8")
        if split == "dev":
            (env / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        else:
            if evaluator_root is None:
                raise ValueError("held-out generation requires evaluator_root")
            private = long_path(evaluator_root) / "manifests"
            private.mkdir(parents=True, exist_ok=True)
            (private / (identifier+".json")).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            (env / "evaluation_ref.json").write_text(json.dumps({"instance_id":identifier,"benchmark_version":VERSION}),encoding="utf-8")
        envs.append(Path(str(env).removeprefix("\\\\?\\")))
    return envs


def semantic_equal(pred, expected):
    if isinstance(expected, bool):
        return pred is expected or (isinstance(pred,str) and pred.strip().lower()==str(expected).lower())
    if isinstance(expected,(int,float)):
        from fsbench.tasks import parse_money
        value = parse_money(pred)
        return value is not None and abs(value-expected)<=.01
    if isinstance(expected,dict):
        return isinstance(pred,dict) and all(k in pred and semantic_equal(pred[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(pred,list) and len(pred)==len(expected) and sorted(map(str,pred))==sorted(map(str,expected))
    return isinstance(pred,str) and " ".join(pred.casefold().split())==" ".join(str(expected).casefold().split())


def score_realistic(answer, gt, workspace):
    if gt.get("output_path"):
        path = workspace / gt["output_path"]
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
            return {"success":False,"score":0.0,"fields":{"file_exists":False}}
        try:
            with path.open(encoding="utf-8-sig",newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader,[])
                rows = list(reader)
            header_ok = [v.strip().casefold() for v in header]==gt["columns"]
            expected = sorted(gt["rows"], key=lambda r:str(r[0]))
            rows = sorted(rows, key=lambda r:str(r[0]) if r else "")
            correct = header_ok and len(rows)==len(expected) and all(len(a)==len(b) and all(semantic_equal(x,y) for x,y in zip(a,b)) for a,b in zip(rows,expected))
            return {"success":correct,"score":float(correct),"fields":{"file_exists":True,"header":header_ok,"rows":correct}}
        except (OSError,UnicodeError,csv.Error):
            return {"success":False,"score":0.0,"fields":{"output_format":False}}
    correct = semantic_equal(answer.get("answer"),gt["answer"])
    return {"success":correct,"score":float(correct),"fields":{"answer":correct}}
