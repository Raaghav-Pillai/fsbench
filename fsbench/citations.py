"""Evidence attribution is scored independently of semantic task correctness."""
import posixpath
import re
from urllib.parse import unquote


def normalize_path(value):
    if isinstance(value, dict):
        value = value.get("path", value.get("source", ""))
    if not isinstance(value, str):
        return ""
    value = value.strip().strip("`\"' ")
    link = re.fullmatch(r"\[[^\]]*\]\(([^)]+)\)", value)
    if link:
        value = link[1]
    value = unquote(value).replace("\\", "/")
    if value.startswith("/workspace/"):
        value = value[len("/workspace/"):]
    elif value.startswith("workspace/"):
        value = value[len("workspace/"):]
    value = re.sub(r"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$", "", value)
    if value.startswith("/") or ".." in value.split("/") or ":" in value:
        return ""
    return posixpath.normpath(value).casefold()


def score_citations(answer, manifest):
    applicable = manifest.get("citation_required", False) or "evidence" in answer or "sources" in answer
    if not applicable:
        return {"evidence_precision": None, "evidence_recall": None, "evidence_attribution_applicable": False}
    raw = answer.get("evidence", answer.get("sources", []))
    if isinstance(raw, str):
        raw = [v.strip(" -\t") for v in raw.splitlines() if v.strip()]
    if not isinstance(raw, list):
        raw = [raw]
    paths = {normalize_path(f["path"]): f["doc_id"] for f in manifest["files"]}
    basenames = {}
    for p, doc in paths.items():
        basenames.setdefault(posixpath.basename(p), set()).add(doc)
    cited, invalid = set(), set()
    for item in raw:
        p = normalize_path(item)
        doc = paths.get(p)
        if not doc and "/" not in p and len(basenames.get(p, ())) == 1:
            doc = next(iter(basenames[p]))
        if doc:
            cited.add(doc)
        else:
            invalid.add(str(item))
    alternatives = manifest.get("acceptable_evidence_sets", [manifest["required_doc_ids"]])
    def match(required):
        required = set(required)
        tp = len(cited & required)
        precision = tp / (len(cited) + len(invalid)) if cited or invalid else 0.0
        recall = tp / len(required) if required else 1.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        return f1, precision, recall
    _, precision, recall = max(map(match, alternatives))
    return {"evidence_precision": precision, "evidence_recall": recall,
            "evidence_attribution_applicable": True, "cited_doc_ids": sorted(cited),
            "invalid_citations": sorted(invalid), "citation_count": len(cited) + len(invalid)}
