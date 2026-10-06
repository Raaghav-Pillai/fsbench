"""Verify actual files in a paired filename-only sweep before spending API credits."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

from fsbench.evaluate import find_envs
from fsbench.paths import long_path


def verify(root, levels=(0, .3, .6, .9)):
    groups = defaultdict(dict)
    for env in find_envs(root):
        m = json.loads((long_path(env) / "manifest.json").read_text(encoding="utf-8"))
        key = (m["task"]["type"], tuple(sorted(m["seeds"].items())))
        level = m["config"]["filename_noise"]
        if level in groups[key]:
            raise ValueError(f"duplicate condition: {key}, {level}")
        groups[key][level] = (env, m)
    if not groups:
        raise ValueError("no environments found")
    files_checked = 0
    fractions = defaultdict(list)
    for key, by_level in groups.items():
        if set(by_level) != set(levels):
            raise ValueError(f"incomplete conditions for {key}: {sorted(by_level)}")
        reference = None
        previous = set()
        clean_paths = None
        for level in sorted(levels):
            env, m = by_level[level]
            config = {k: v for k, v in m["config"].items() if k != "filename_noise"}
            files, paths, renamed = {}, set(), set()
            current_paths = {}
            for f in m["files"]:
                identity = (f["doc_id"], f["copy"])
                path = PurePosixPath(f["path"])
                digest = hashlib.sha256((long_path(env) / "workspace" / f["path"]).read_bytes()).hexdigest()
                if digest != f["sha256"] or path.suffix != "." + f["format"]:
                    raise ValueError(f"invalid file: {env}, {path}")
                if f["path"].casefold() in paths or identity in files:
                    raise ValueError(f"duplicate path or identity: {env}, {path}")
                paths.add(f["path"].casefold())
                files[identity] = (str(path.parent), f["format"], f["role"], digest)
                current_paths[identity] = f["path"]
                if f["noisy_name"]:
                    renamed.add(identity)
                files_checked += 1
            snapshot = (config, m["task"], m["ground_truth"], m["required_doc_ids"],
                        m["trap_doc_ids"], m["decoy_answers"], m["oracle"], files)
            if reference is None:
                reference, clean_paths = snapshot, current_paths
            if snapshot != reference:
                raise ValueError(f"non-filename change for {key} at {level}")
            if 0 in levels and renamed != {i for i in files if current_paths[i] != clean_paths[i]}:
                raise ValueError(f"renaming flags do not match paths: {env}")
            if not previous <= renamed:
                raise ValueError(f"non-monotonic selection: {env}")
            previous = renamed
            fractions[level].append(len(renamed) / len(files))
    return {"paired_seed_groups": len(groups), "environments": len(groups) * len(levels),
            "files_checked": files_checked,
            "mean_fraction_renamed": {str(k): sum(v) / len(v) for k, v in sorted(fractions.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
