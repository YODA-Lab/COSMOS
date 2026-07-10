# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper *COSMOS: Model-Agnostic Personalized Federated Learning with Clustered Server Models and Pseudo-Label-Only Communication*. See [README.md](README.md) for the public overview.

**Critical naming fact:** the paper's method **COSMOS** is called **`MAPL`** everywhere in the code (`AlgorithmSelected.MAPL`). Treat "COSMOS" and "MAPL" as the same thing.

This is research-grade code, not a package: no `src/` layout, no `setup.py`, no tests, no CLI. Expect hard-coded experiment sweeps, commented-out alternatives, and a very large `entities.py` (~225 KB).

## Commands

```bash
# Run experiments (edit the config first — see below)
python main_.py

# Generate paper figures/CSVs from results/ into figures/
python read_json.py

# Install deps: GPU (CUDA 12.1) or CPU
pip install -r requirements.txt      # requirements.txt is UTF-16 encoded
pip install -r new_require.txt       # CPU alternative
```

There is no build step, no linter config, and no test suite.

## How experiments are configured (important)

**All configuration lives in code, not on the command line.** To change what runs, edit the
`if __name__ == '__main__':` block at the bottom of [main_.py](main_.py) (around lines 683–821).
You set Python lists there: `data_sets_list`, `num_clients_list`, `algorithm_selection_list`,
`nets_types_list_PseudoLabelsClusters`, `alpha_dichts`, `cluster_technique_list`, etc. The block
runs nested loops over these and calls `run_exp_by_algo()`, which dispatches to the per-algorithm
`run_*` function based on `experiment_config.algorithm_selection`.

Hyperparameter defaults live in `ExperimentConfig.__init__` in [config.py](config.py); the
`__main__` block overrides them per run.

## Architecture

The whole project shares one global mutable config object:

- **`experiment_config`** — a singleton instance of `ExperimentConfig`, defined at the end of
  [config.py](config.py) and imported everywhere via `from config import *`. Nearly every function
  reads hyperparameters off this global. `update_net_type()` and `update_num_classes()` mutate it to
  set per-architecture learning rates and per-dataset class counts. When tracing behavior, check what
  has been written to `experiment_config` upstream — state is threaded implicitly, not by arguments.

Four core files:

- **[config.py](config.py)** — enums (`DataSet`, `AlgorithmSelected`, `NetType`, `NetsType`,
  `ClusterTechnique`, `NetClusterTechnique`, `ServerFeedbackTechnique`, …) plus `ExperimentConfig`.
  The `NetsType` enum encodes a *client+server architecture pair* (e.g. `C_alex_S_vgg`).

- **[entities.py](entities.py)** — the heart of the project. Contains:
  - Model classes: `AlexNet`, `VGGServer`, `ResNet18Server`, `MobileNetV2Server`,
    `SqueezeNetServer`, `DenseNetServer`. Each supports a `num_clusters` multi-head option and
    `forward(x, cluster_id=None)`.
  - `LearningEntity` (base) → `Client` and `Server` for COSMOS/MAPL, plus a `Client_*`/`Server_*`
    pair per baseline algorithm.
  - **The COSMOS clustering logic lives in the `Server` class**: `greedy_elimination` (the paper's
    Algorithm 1), `k_means_grouping`, `manual_grouping`, distance functions (`calc_L2`,
    `calc_cross_entropy`, `compute_distances`), and feedback creation
    (`create_feed_back_to_clients_multihead` / `_multimodel`).

- **[functions.py](functions.py)** — data pipeline and result saving. `create_data()` →
  `get_data_set()` downloads/splits datasets; Dirichlet non-IID partitioning
  (`get_image_split_list_classes_dich`); `create_clients()` instantiates the correct `Client_*`
  subclass for the selected algorithm; `save_record_to_results()` writes results.

- **[main_.py](main_.py)** — entry point and federated-learning loops. `RecordData` collects
  `experiment_config.to_dict()` plus per-client/server top-1/5/10/100 accuracies and message sizes,
  and is serialized to one JSON per seed under `results/<descriptive-folder>/`.

## Data flow

`main_.py __main__` sets `experiment_config` → `create_data()` splits data (client shards + a public
server set) → `create_clients()` builds the entities → the FL loop alternates `client.iterate(t)` and
`server.iterate(t)`, exchanging **pseudo-labels only** (no weights/gradients) → `RecordData` +
`save_record_to_results()` write JSON → `read_json.py` turns JSON into figures/CSVs.

## Gotchas

- `entities.py` is huge; use symbol search (find the `class`/`def`) rather than reading top-to-bottom.
- Two divergent requirements files exist (GPU vs CPU); `requirements.txt` is UTF-16.
- `read_json.py` is the current analysis pipeline for turning results into figures/CSVs.
- Git history messages are uninformative ("before changing top5"); don't rely on them for intent.
