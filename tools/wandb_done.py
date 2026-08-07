#!/usr/bin/env python
"""Is there a FINISHED W&B run by this name in this project? Exit 0 yes / 1 no / 2 unknown.

The sweep's cross-machine skip ledger: completion is recorded in W&B by the training run
itself, so any machine can consult it -- no status files to sync between boxes. rsl_rl
prefixes its own timestamp to the display name, so the match is anchored at the suffix.

Usage: wandb_done.py <project> <run_name>
"""

import sys

def main() -> int:
    project, run_name = sys.argv[1], sys.argv[2]
    try:
        import wandb

        api = wandb.Api(timeout=15)
        entity = api.default_entity
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": {"$regex": f".*{run_name}$"}, "state": "finished"},
            per_page=3,
        )
        return 0 if next(iter(runs), None) is not None else 1
    except Exception as exc:  # offline, no credentials, project absent, API change ...
        print(f"wandb_done: query failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
