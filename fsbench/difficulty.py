"""Verify facts and evidence identities across composed interventions."""
import json
from collections import defaultdict

from fsbench.paths import long_path


def verify_composed(envs) -> dict:
    groups = defaultdict(list)
    for env in envs:
        m = json.loads((long_path(env) / "manifest.json").read_text(encoding="utf-8"))
        groups[(m["task"]["type"], m["seeds"]["world"], m["seeds"]["task"])].append(m)
    for key, manifests in groups.items():
        first = manifests[0]
        identities = {}
        for m in manifests:
            for field in ("task", "ground_truth", "required_doc_ids"):
                if m[field] != first[field]:
                    raise ValueError(f"composed intervention changed {field} for {key}")
            for f in m["files"]:
                digest = f.get("document_sha256")
                if digest is None:
                    raise ValueError("composed integrity requires document_sha256; regenerate these environments")
                prior = identities.setdefault(f["doc_id"], digest)
                if digest != prior:
                    raise ValueError(f"document contents changed: {f['doc_id']}")
    return {"worlds": len(groups), "environments": sum(map(len, groups.values()))}
