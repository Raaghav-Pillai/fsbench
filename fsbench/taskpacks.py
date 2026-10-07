"""Build versioned candidate packs; hold-out seeds and labels stay evaluator-side."""
import hashlib
import json
import secrets
from pathlib import Path

from fsbench.realistic import VERSION, DOMAINS, LEVELS, generate_realistic


def recipe_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("realistic.py", "docs.py", "render.py", "rng.py"):
        digest.update(name.encode()+b"\0"+(root/name).read_bytes().replace(b"\r\n",b"\n"))
    return digest.hexdigest()


def check_version(version):
    if version != VERSION:
        raise ValueError(f"unknown task pack {version}")
    lock = Path(__file__).parent / "packs" / (version+".json")
    if not lock.is_file():
        raise ValueError("benchmark version lock is missing")
    spec = json.loads(lock.read_text(encoding="utf-8"))
    if spec["recipe_sha256"] != recipe_digest():
        raise ValueError("task data recipe changed; assign a new benchmark version before building")
    return spec


def build_pack(out, *, version=VERSION, num_worlds=100, seed_file=None, test_only=False, exclude_root=None):
    spec = check_version(version)
    if num_worlds < 1:
        raise ValueError("num_worlds must be positive")
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("benchmark output must be empty; immutable datasets are not overwritten")
    seeds = json.loads(Path(seed_file).read_text(encoding="utf-8")) if seed_file else [secrets.randbits(128) for _ in range(num_worlds)]
    if not isinstance(seeds,list) or len(seeds)!=num_worlds or any(type(s) is not int or s<0 for s in seeds) or len(set(seeds))!=num_worlds:
        raise ValueError("seed file must be a list of num_worlds distinct nonnegative integers")
    if exclude_root:
        previous = json.loads((Path(exclude_root)/"benchmark.json").read_text(encoding="utf-8"))
        used = {row["world_id"] for row in previous["instances"]}
        proposed = {hashlib.sha256(f"{version}\0{DOMAINS[i%len(DOMAINS)]}\0{seed}".encode()).hexdigest()[:24] for i,seed in enumerate(seeds)}
        if proposed & used:
            raise ValueError("test worlds overlap the excluded benchmark; supply fresh private seeds")
    root.mkdir(parents=True,exist_ok=True)
    private = root / ".private"
    private.mkdir()
    (private / "seeds.json").write_text(json.dumps(seeds),encoding="utf-8")
    index = []
    for i,seed in enumerate(seeds):
        domain = DOMAINS[i%len(DOMAINS)]
        # Split by world, never by task; all five sibling tasks stay together.
        fraction = i/num_worlds
        split = "test" if test_only else "dev" if fraction<.6 else "validation" if fraction<.8 else "test"
        level = LEVELS[(i//len(DOMAINS))%len(LEVELS)]
        envs = generate_realistic(domain,seed,root/split,level=level,split=split,evaluator_root=private)
        from fsbench.benchmark import load_manifest
        for env in envs:
            m = load_manifest(env)
            index.append({"instance_id":m["env_id"],"world_id":m["world_id"],"split":split,
                "domain":domain,"template":m["task_template"],"families":m["task_families"],"difficulty_level":level,
                "path":env.relative_to(root.resolve()).as_posix() if env.is_absolute() else env.relative_to(root).as_posix()})
    dataset = {"benchmark_family":"realistic","benchmark_version":version,"status":"candidate; human validation pending",
        "recipe_sha256":spec["recipe_sha256"],"worlds":num_worlds,"tasks":len(index),"instances":index}
    (root / "benchmark.json").write_text(json.dumps(dataset,indent=2),encoding="utf-8")
    return dataset
