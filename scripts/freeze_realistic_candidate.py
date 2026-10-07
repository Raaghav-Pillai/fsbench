"""Create (never overwrite) a candidate version lock after task authoring."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fsbench.realistic import VERSION, DOMAINS, world_material
from fsbench.taskpacks import recipe_digest

root = Path(__file__).resolve().parents[1]/"fsbench"/"packs"
root.mkdir(exist_ok=True)
spec = {"benchmark_version":VERSION,"status":"candidate; human validation pending","recipe_sha256":recipe_digest(),
        "domains":list(DOMAINS),"templates":[{"id":t["template"],"capabilities":t["capabilities"]}
            for d in DOMAINS for t in world_material(d,0)[1]]}
with (root/(VERSION+".json")).open("x",encoding="utf-8") as fh:
    json.dump(spec,fh,indent=2)
print(f"Created immutable candidate lock for {VERSION}")
