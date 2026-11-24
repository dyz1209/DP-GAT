# DP-GAT Preprocessing and Data Splitting Code

This folder contains modules for FFT preprocessing, adaptive average pooling, nearest-neighbor search, graph construction, and data splitting.

## Files

| File | Function |
| --- | --- |
| `fft_preprocessing.py` | Non-DC FFT amplitude spectra and standardization using training-set statistics |
| `adaptive_pooling.py` | Adaptive average pooling after standardization |
| `nearest_neighbor_search.py` | Euclidean distance search and union symmetrization of the training KNN graph |
| `graph_construction.py` | Training graph construction, validation/test node attachment, and RBF edge weights |
| `data_split.py` | Window extraction, chronological splitting, gap removal, and labeled training-node selection |
| `run_prepare.py` | Integration of the modules and export of inputs and split results |

## Processing Sequence

1. Each CSV column contains a signal sequence for a different class. Missing values are removed from each column, and non-overlapping windows of the specified length are extracted.
2. Within each class, windows are divided chronologically into 60% training, a discarded 5% gap, 10% validation, a discarded 5% gap, and the remaining windows for testing. Training, gap, and validation counts are calculated using Python's `round` function.
3. Single-sided FFT amplitude spectra are extracted with the DC component removed. The mean and standard deviation of each frequency dimension are fitted using training nodes only, and the same statistics are applied to all partitions.
4. Adaptive average pooling reduces each standardized spectrum to 64 dimensions.
5. Self-neighbors are excluded from the training graph. The default is K=5. Neighbor edges and their reverse edges are combined by set union and deduplicated.
6. Validation and test nodes are attached separately through five training anchors per query node. Edges are directed from training nodes to query nodes. By default, no reverse edges or edges between query nodes are created. Candidate edges exclude self-loops; self-loops are handled by subsequent model operations.
7. The random seed controls class-wise selection of labeled nodes within the training partition. The default label rate is 20%, with at least one labeled node per class. The fixed chronological split does not change across seeds. Graph functions do not receive class labels.

## Dataset Presets

| Dataset | Window length | Original windows per class | Training / validation / test per class | Discarded per class |
| --- | ---: | ---: | --- | ---: |
| A | 1024 | 1023 | 614 / 102 / 205 | 102 |
| B | 1024 | 1023 | 614 / 102 / 205 | 102 |
| C | 256 | 62 | 37 / 6 / 13 | 6 |
| D | 1024 | 200 | 120 / 20 / 40 | 20 |

## Experimental Environment

The experiments use 64-bit Windows, with the Python environment managed through Anaconda. Computation is performed on an Intel Core i5-1155G7 CPU @ 2.50 GHz with 16 GB of RAM. The entry point in this folder uses four CPU computation threads by default; this can be adjusted using `--cpu-threads`.

| Software or dependency | Version | Purpose |
| --- | --- | --- |
| Python | 3.9.25 | Program runtime |
| PyTorch | 2.8.0 | FFT, standardization, adaptive pooling, and distance computation |
| NumPy | 1.26.4 | Array and numerical processing |
| pandas | 2.3.3 | CSV loading and data processing |
| PyTorch Geometric | 2.6.1 | Graph neural network layers in the complete model |
| scikit-learn | 1.6.1 | Classification metrics in the complete experiments |
| Matplotlib | 3.5.1 | Figure generation in the complete experiments |
| psutil | 7.1.3 | Process memory measurement in the complete experiments |

Install the dependencies required to run the code in this folder:

```text
python -m pip install torch==2.8.0 numpy==1.26.4 pandas==2.3.3
```

## Running the Code

Datasets A, B, C, and D are loaded by default from `dataset_A.csv`, `dataset_B.csv`, `dataset_C.csv`, and `dataset_D.csv`, respectively, in the parent directory of this folder. Use `--csv` to specify a data file. Output metadata identifies the input dataset by its letter only and does not record the input filename or path.

Run the following commands from a terminal in this folder:

```text
python run_prepare.py --dataset A --seed 42
python run_prepare.py --dataset B --seed 42
python run_prepare.py --dataset C --seed 42
python run_prepare.py --dataset D --seed 42
```

Use `--csv`, `--step`, `--nodes-per-class`, `--pooling-dim`, and `--output-dir` to override the corresponding settings.

Outputs are saved under `outputs/<dataset>_seed_<seed>/`:

- `prepared_inputs.pt`: Features, labels, masks, labeled training nodes, standardization statistics, original window indices within each class, and the three candidate graphs.
- `split_metadata.json`: Parameter settings, counts, index conventions, and environment information.

The first row of `edge_index` contains source nodes, and the second row contains target nodes. The training graph contains training nodes only. Node order in the validation/test graphs is `[training nodes, corresponding query nodes]`. Original window indices within each class start at zero, with signal intervals `[index * step, (index + 1) * step)`.

The training graph is used for model training, the validation graph for validation, and the test graph for final evaluation.
