# Contributing to VehicleMind

VehicleMind uses short-lived feature branches and pull requests. Do not develop directly on `main`.

## Development setup

```bash
uv sync --group dev
uv run --group dev python -m pytest -q
uv run --group dev ruff check apps modules scripts tests
uv run --group dev ruff format --check apps modules scripts tests
uv run --group dev python scripts/check_source_size.py
```

Install the perception stack only for camera, video, or model work:

```bash
uv sync --extra perception --group dev
```

Python 3.13 is required. The current full perception path is validated on macOS 26.6.2, Apple Silicon; the core test suite is designed to remain camera-, weight-, and API-free.

## Pull requests

1. Create a branch with a focused name, such as `feature/replay-evaluation` or `fix/presence-contract`.
2. Add or update tests before implementation when behavior changes.
3. Keep public contracts backward-compatible or document the migration.
4. Run the commands above and include reproducible evidence in the PR description.
5. Do not mark a `todolist.md` item complete until its acceptance condition is verifiably met.

Formal evaluations and reproducible replays must create a clean-worktree run record before processing data:

```bash
uv run --group dev python scripts/snapshot_experiment_config.py \
  --run-id <descriptive-run-id> \
  --output-root runs \
  --asset-id <manifest-asset-id>
```

Keep both `resolved_config.yaml` and `run_card.md` with the result package. The run card is provenance evidence, not a substitute for measured accuracy, latency, robustness, or qualitative examples.

Production modules are limited to 500 physical lines. Executable `*_demo.py` files are limited to 300 lines. Extract code by responsibility rather than bypassing the check.

## Assets, data, and secrets

- Never commit API keys, `.env`, private data, raw datasets, or model weights.
- Record every model in `assets/model_manifest.yaml`, including its source, hash, license status, and expected path.
- Run `python scripts/verify_assets.py` before perception demos.
- Do not add third-party media or weights until redistribution permission is documented.
- Preserve subject-level train/validation/test isolation for all driver-monitoring experiments.

## Commit style

Use imperative, scoped messages such as `fix: preserve unknown eye observations` or `test: add replay safety cases`. Keep formatting-only changes separate from behavior changes when practical.
