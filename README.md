# Personalized Federated Heterogeneous Graph Learning for Credit Risk

## Tài liệu theo thư mục

Các README tiếng Việt mô tả vai trò, cách sử dụng và trạng thái triển khai:

- [Scripts và thứ tự chạy pipeline](scripts/README.md)
- [Mã nguồn và các module](src/README.md)
- [Cấu hình](configs/README.md)
- [Dữ liệu và vòng đời các đầu ra](data/README.md)
- [Notebook phân tích](notebooks/README.md)
- [Kết quả thí nghiệm](results/README.md)
- [Kiểm thử](tests/README.md)

## Experiment protocol

1. Aggregate bureau_balance into bureau, then drop columns with >80% missing
   globally. This existing preparation step is unchanged. Keep relation keys.
2. Allocate customers to N clients using label-based Dirichlet sampling.
3. Split each client locally by TARGET into train/test (default 80/20).
4. Fit ONE shared encoder/scaler on the union of local-train rows only, including
   history rows owned by those train customers. Never fit on local-test rows.
5. Transform all clients with that fitted encoder. Save one heterogeneous graph
   file per client, with Customer train_mask/test_mask.
6. Train FedAvg and independent local-only GNNs using train customers only.
7. Log each client's local-test AUC and pooled test AUC for both methods per round.
8. Sweep N=10,20,30,40,50; fit a fresh shared transformer for each N/seed split.

This is a centralized simulation of FL with shared train-only preprocessing.
It does not implement a privacy-preserving distributed encoder fit. Global
missing selection is kept as requested; a strict held-out protocol would also
learn that selection mask using training data only.

## Setup and commands

Use the project virtual environment (PyTorch, pandas, NumPy, scikit-learn):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Preparation was already performed in data/interim/tables/. To regenerate it:

```powershell
python scripts/01_prepare_tables.py
```

Run one experiment:

```powershell
python scripts/02_partition_clients.py --num-clients 10 --alpha 0.5 --seed 42
python scripts/02b_fit_encoder.py
python scripts/03_build_graphs.py
python scripts/04_train_federated.py --rounds 20 --local-epochs 1 --batch-size 256
python scripts/05_evaluate.py
```

### Train independent GNNs per client

After partitioning, fitting the shared encoder, and building the v2 encoded
graphs, run the dedicated local-only trainer:

```powershell
python scripts/04_train_local_gnn.py --epochs 20 --batch-size 128
python scripts/04_train_local_gnn.py --clients client_000 client_001 --epochs 20 --device cpu
```

It uses the existing relation-specific mean GNN (2 layers, hidden size 32), one
independent model/Adam optimizer per client, and the common feature encoder.
Customer binary classification uses unweighted BCEWithLogitsLoss on train_mask
only. Complete customer components form batches, preserving normal and orphan
relations. The final epoch is saved; test metrics never select checkpoints.
No validation split or early stopping is introduced. Average precision is
reported alongside ROC AUC; either is blank for a single-class/empty test set.
Clients without train customers are marked skipped instead of evaluating an
untrained model. Device auto-selects CUDA if available, otherwise CPU.

Outputs in results/local_gnn/:
- run_config.json and summary.csv across the selected clients
- client_XXX/model.pt (weights, architecture, encoder/schema identity)
- client_XXX/history.csv (epoch, train loss, optimizer steps, local test metrics)
- client_XXX/test_predictions.csv (SK_ID_CURR, TARGET, probability_default)

Only one client graph is loaded at a time, but its encoded arrays must fit RAM.
The batch-size option limits customer components used for each forward/backward
pass; it does not cap neighbors per customer or reduce graph-loading memory.
The supplied RelBench examples use their own task/database wrappers; this entry
point consumes this project's six-type .npz graphs and TARGET labels directly.
Model code: src/models/hetero_gnn.py. Training: src/models/local_training.py.

Run the requested sweep directly from prepared tables (steps 2-7 are automatic):

```powershell
python scripts/06_sweep_clients.py --client-counts 10 20 30 40 50 --alpha 0.5 --seeds 42 --rounds 20
```

Optional multiple seeds: `--seeds 42 43 44`. Use `--device cuda` only with a CUDA
PyTorch installation. The default is CPU. The scripts use CLI defaults; YAML
files document defaults and are not automatically loaded. The old
01_preprocess.py is an application-only utility outside this experiment flow.

## Partition details

For each TARGET class, reserve min_per_class samples for every client, then
allocate the remaining class samples using a Dirichlet(alpha) vector and a
multinomial draw. The default minimum is 2: this is a CONSTRAINED Dirichlet
partition that supports both classes in local train/test. It is not an
unconditioned Dirichlet sample. Set --min-per-class 0 for unconditioned allocation;
clients can then be empty or single-class and AUC can be undefined. Invalid
requests with too few class samples fail explicitly. The previous semantic
region/occupation partition remains available via --strategy semantic.

The local stratified split rounds test counts separately for each class and
keeps at least one train and one test sample when a class has at least 2 samples.
Singletons remain train. Thus small clients can deviate from exactly 80/20.
No oversampling, undersampling, or class weighting is applied. Counts by class
and split are recorded in partition_report.json. Customer IDs are disjoint
between clients; related rows follow SK_ID_CURR. Rows outside application_train
remain in the prepared sources and are reported as unassigned.

## Shared features and graphs

Shared encoder statistics and vocabularies are accumulated across all clients'
train-owned rows. Numeric values use pooled train mean imputation and standard
deviation scaling. All-missing train numeric columns remain present (zero).
Categorical values use a shared one-hot vocabulary with explicit missing and
unknown slots. IDs and TARGET are excluded from features. The fitted JSON stores
feature order, dimensions, fit row counts, split hashes and a fingerprint.
Encoder fitting is independent of client/test values; no columns are dropped
again. Both comparison methods use this same preprocessing protocol.

Each client .npz graph contains x__<node>, owner__<node>, mapping__<node>__<key>,
edge__<relation>, y, customer_ids, train_mask, test_mask, and JSON metadata.
Six node types: customer, bureau, previous_application, installment, pos_cash,
credit_card. Schema `orphan_fallback_v2` keeps eight relation types plus reverses,
including empty edge arrays so every client has the same relation schema:

```text
customer --has--> bureau / previous_application
previous_application --has--> installment / pos_cash / credit_card (parent exists)
customer --has_orphan--> installment / pos_cash / credit_card (parent absent only)
```

A transaction has exactly one forward parent edge. There is no direct Customer
shortcut for transactions with a valid Previous parent. `is_orphan_prev` is
appended as the LAST unscaled binary feature of all three transaction types,
including clients without orphans, and is also saved as a separate graph array.
The shared encoder stays unchanged; this structural flag is computed at graph
build time without labels. Extra feature metadata and a graph schema version
are saved to prevent mixing old/new graphs.

Transactions whose SK_ID_CURR is null/absent from this client, or whose existing
Previous belongs to another Customer, are excluded from graph nodes and saved
to graphs/quarantine/client_XXX/<table>.csv with source_row_id and a reason.
Quarantine counts/reasons are in graph metadata (or graph_report.json in topology
mode). Source tables are never modified. Node IDs are rebuilt contiguously;
source_row_id preserves the original zero-based input row position after filtering.
Invalid Customer/Bureau/Previous entity keys still raise errors.
Bureau monthly history is aggregated, not a seventh node type.

Rebuild ALL client graphs after this schema change, using 03_build_graphs.py
after partition and encoder fitting are complete. Retrain models; old checkpoint
input dimensions and relation parameters are not compatible. No need to refit
an existing encoder solely for the added structural flag if splits are unchanged.

Customer components have no cross-customer edges. Training batches contain only
train-customer components, including all of their history nodes. Test components
are used only for prediction. The GNN has no batch normalization or operation
that mixes disconnected customers' statistics. Keeping both masks in one file
does not make test labels or test features participate in training.

## Models, metrics and outputs

The implemented baseline is a relation-specific mean-message-passing GNN,
2 layers and hidden size 32 by default, followed by a Customer binary head.
All-client FedAvg weights client updates by the number of train customers.
Local-only models use the same architecture, initial weights, shared encoder,
local splits and total local epochs. Local-only Adam state persists; FedAvg
local Adam is reset each round. This is a FedAvg baseline, not an implementation
of a specialized personalization algorithm or an exact reproduction of a paper.

Binary cross-entropy is unweighted and uses only train customers. Evaluation is
reported at every round with a fixed final-round checkpoint; no test-based best
checkpoint selection or hyperparameter search is performed. Undefined AUC is
blank in CSV/null in JSON. Pooled AUC is calculated on concatenated predictions,
NOT the mean of local AUCs. Pooled local-only scores come from different models
and may have different calibration. The pooled readout is not an independently
held-out external test set. Across N, local splits change; use multiple seeds
for research conclusions rather than assuming the pooled test IDs are fixed.

Single run outputs:
- data/processed/clients/client_XXX/: six CSVs + customer_split.csv
- data/processed/clients/partition_report.json and assignments.csv
- data/processed/shared_encoder.json
- data/processed/graphs/client_XXX.npz + manifest.json
- results/default/metrics.csv (round, client, method, auc, test counts)
- results/default/federated.pt, local_models/*.pt and run_config.json
- results/default/final_evaluation.csv from the evaluation command

Sweep outputs are isolated in results/sweep/n_NN_seed_SEED/. Combined
results/sweep/sweep_metrics.csv adds num_clients, alpha and seed to every local
and pooled metric row. Full sweeps retain prepared client tables, graphs and
checkpoints for each run and require substantial disk space and compute.

## Validation

```powershell
python -m unittest discover -s tests -p test_prepare_tables.py -v
python -m unittest discover -s tests -p test_federated_pipeline.py -v
python -m unittest discover -s tests -p test_dirichlet_training.py -v
```

Tests include train/test isolation for encoder fitting and gradient updates,
Dirichlet repeatability, ownership, reverse edges, checkpoint evaluation, and a
small end-to-end sweep. Small synthetic AUCs are software checks, not research
results on Home Credit.


### ROC-AUC and PR-AUC

All evaluation entry points log `roc_auc` and `pr_auc` on Customer test masks.
PR-AUC is trapezoidal area under the precision-recall curve, with TARGET=1
as the positive class. `average_precision` remains a separate metric and
`auc` remains a compatibility alias for ROC-AUC. Empty or single-class test
sets have null metrics (blank CSV cells). Pooled metrics are computed from
concatenated test predictions, not averaged client metrics. Existing CSVs
are not automatically updated; subsequent runs produce the new columns.
