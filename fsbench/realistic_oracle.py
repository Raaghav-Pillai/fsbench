"""Independent content-derived oracle for the authored templates.

The evaluator supplies source identities, but answers are computed from extracted
visible text, not copied from ground_truth. This is a solvability check, not a human baseline.
"""
import csv
import io
import json
import re
from statistics import median


def field(text, name, default=None):
    if text.lstrip().startswith("{"):
        return json.loads(text).get("fields", {}).get(name, default)
    match = re.search(r"^"+re.escape(name)+r"\s*[:\t]\s*(.+)$", text, re.M)
    return match[1].strip() if match else default


def table(text, first_column):
    lines = text.splitlines()
    for i,line in enumerate(lines):
        delim = "\t" if "\t" in line else ","
        cols = next(csv.reader([line], delimiter=delim))
        if cols and cols[0].strip() == first_column:
            return list(csv.DictReader(lines[i:], delimiter=delim))
    return []


def solve(manifest, tools):
    from fsbench.runner import OracleAgent
    texts, sources = {}, []
    for key in manifest["required_doc_ids"]:
        record = next(f for f in manifest["files"] if f["doc_id"]==key and f["copy"]==0)
        path = record["path"]
        sources.append(path)
        texts[key] = tools.call(OracleAgent.READ_TOOL[tools.name], {"path":"/workspace/"+path})
        if texts[key].startswith("Error:"):
            raise ValueError(texts[key])
    domain, name = manifest["task_template"].split(".")
    get = lambda key, prop, default=None: field(texts.get(key,""),prop,default)
    output_rows = None
    if domain == "finance":
        invoices = table(texts["invoices"],"invoice_id")
        payments = table(texts.get("payments",""),"invoice_id")
        paid = {}
        for r in payments:
            paid[r["invoice_id"]] = paid.get(r["invoice_id"],0)+float(r["amount"])
        credit = float(get("credit","Approved credit",0))
        due = [[r["invoice_id"], float(r["amount"])-paid.get(r["invoice_id"],0)
                -(credit if r["invoice_id"]==get("credit","Applies to") else 0)] for r in invoices if r["status"]=="posted"]
        if name == "unpaid":
            answer = [r[0] for r in due if r[1]>0]
        elif name == "balance":
            answer = sum(r[1] for r in due)
        elif name == "latest_invoice":
            answer = max((r for r in invoices if r["status"]=="posted" and r["issued"]<="2026-09-30"),key=lambda r:r["issued"])["invoice_id"]
        elif name == "compliance":
            answer = sum(r[1] for r in due)<=0
        else:
            output_rows = [r for r in due if r[1]>0]
            answer = "output/payment_summary.csv"
    elif domain == "legal":
        value = float(get("amendment","Annual value",get("signed","Annual value",0)))
        agreement = get("signed","Agreement")
        if name == "current_terms":
            answer = {"agreement":agreement,"annual_value":value}
        elif name == "renewal":
            answer = get("amendment","Renewal date",get("signed","Renewal date"))
        elif name == "compare_terms":
            answer = value-float(get("prior","Annual value"))
        elif name == "authority":
            issued = [r for r in table(texts["register"],"agreement") if r["state"]=="in force"]
            answer = issued[0]["agreement"]
            if answer != agreement:
                raise ValueError("execution register and signed agreement disagree")
        else:
            answer = value
    elif domain == "hr":
        if name in ("manager","manager_provenance"):
            answer = get("transfer","Manager",get("employee","Manager"))
        elif name == "payroll":
            pay = {r["component"]:float(r["amount"]) for r in table(texts["payroll"],"component")}
            answer = round(float(get("employee","Annual salary"))/12,2)+pay["approved bonus"]
        elif name == "latest_review":
            answer = get("review","Rating")
        else:
            rows = table(texts["checklist"],"employee")
            output_rows = [[r["employee"],r["item"]] for r in rows if r["complete"]=="no"]
            answer = "output/onboarding.csv"
    elif domain == "research":
        sales, refunds = table(texts.get("sales",""),"month"), table(texts.get("refunds",""),"month")
        if name in ("claim_provenance","reconstruct"):
            answer = sum(float(r["sales"]) for r in sales if r["month"] in ("7","8","9"))-sum(float(r["refunds"]) for r in refunds if r["month"] in ("7","8","9"))
            if name=="claim_provenance" and answer!=float(get("report","Net revenue")):
                raise ValueError("report arithmetic disagrees with sources")
        elif name == "latest_analysis":
            rows = table(texts["registry"],"study")
            answer = int(next(r["version"] for r in rows if r["status"]=="issued"))
            if answer!=int(get("report","Version")):
                raise ValueError("publication register disagrees with note")
        elif name == "compare_estimate":
            answer = float(get("report","Net revenue"))-float(get("previous","Net revenue"))
        else:
            answer = all({r["month"] for r in rows}>={"7","8","9"} for rows in (sales,refunds))
    else:
        results = [r for r in table(texts.get("results",""),"run_id") if r["status"]=="complete"]
        if name == "latest_result":
            answer = max(results,key=lambda r:r["completed"])["run_id"]
            if answer!=get("release","Accepted run"):
                raise ValueError("latest completed run not accepted")
        elif name == "config_provenance":
            answer = int(get("config","batch_size"))
            if get("release","Accepted run") not in texts["log"] or "config="+get("config","Config ID") not in texts["log"]:
                raise ValueError("configuration provenance chain broken")
        else:
            by_config = {r["config"]:float(r["median_ms"]) for r in results}
            if "log" in texts:
                samples = re.search(r"samples_ms=([\d,]+)",texts["log"])[1]
                if median(map(float,samples.split(",")))!=by_config["candidate"]:
                    raise ValueError("summary and raw log disagree")
            if name == "compare_runs":
                answer = by_config["candidate"]-by_config["baseline"]
            elif name == "regression":
                answer = by_config["candidate"]>1.1*by_config["baseline"]
            else:
                answer = "output/benchmark_summary.csv"
                output_rows = [[r["run_id"],float(r["median_ms"])] for r in results]
    if output_rows is not None:
        columns = {"finance":["invoice_id","amount_due"],"hr":["employee","item"],"engineering":["run_id","median_ms"]}[domain]
        stream = io.StringIO()
        writer = csv.writer(stream,lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(output_rows)
        tools.call("write_file", {"path":"/workspace/"+answer,"content":stream.getvalue()})
    return {"answer":answer,"evidence":sources}
