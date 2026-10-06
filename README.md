# FS-EntropyBench

A benchmark for measuring how filesystem organization affects AI agents' ability to complete tasks. It generates synthetic company file shares and varies folder depth, file names, clutter, duplicate files, outdated versions, and document formats while keeping the underlying task and answer fixed.

Agents retrieve information, reconcile records, resolve conflicting documents, and produce output files. The benchmark tracks task success, tool calls, tokens, cost, and latency.

Blog post: https://raaghav-pillai.github.io/benchmarks/fsbench/

## Quick start

Requires Python 3.10+. Clone the repository, create and activate a virtual environment, then install:

```sh
git clone https://github.com/Raaghav-Pillai/file_benchmarking.git
cd file_benchmarking
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then run:

```sh
python -m pip install -e ".[dev]"
fsbench generate --task retrieve --preset organized --seed 0 --out envs/demo
fsbench oracle --envs envs/demo --out runs/demo-oracle
fsbench evaluate --runs runs/demo-oracle --out results-demo.csv
```

The oracle uses the hidden ground truth to sanity-check the environment; it is not a model performance result. To run a model, set `OPENROUTER_API_KEY` in your environment or a local `.env` file, then use:

```sh
fsbench run --envs envs/demo --model YOUR_PROVIDER/YOUR_MODEL --out runs/demo-model
fsbench evaluate --runs runs/demo-model --out results-model.csv
```

Use `fsbench --help` for commands, including `sweep` for controlled experiments and `plot` for charts (requires `python -m pip install -e ".[plot]"`).

For controlled filename experiments, see the [filename noise guide](docs/filename-noise.md), including the paired 80-run recipe and diagnostic definitions.

## Help out

Contributions are welcome: add realistic tasks, improve file layouts and distractors, extend agent integrations, or run reproducible experiments.

1. Fork the repository and create a branch for your change.
2. Follow the setup above and try the demo to understand the workflow.
3. Make a focused change. Start with `fsbench/tasks.py` for tasks, `fsbench/config.py` and `fsbench/layout.py` for filesystem conditions, or `fsbench/evaluate.py` for metrics.
4. Add relevant tests for code changes and run `python -m pytest`.
5. Open a pull request explaining what changed and how you checked it. For experiments, include the model, commands, seeds, conditions, and a results summary. Keep API keys and generated workspaces out of commits.

You can also [open an issue](https://github.com/Raaghav-Pillai/file_benchmarking/issues) with an idea, bug report, or unexpected result.
