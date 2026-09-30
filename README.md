# Hierarchical Graph Neural Network for Interpretable Facial Weakness Assessment in FSHD

Research code for the paper **“Hierarchical Graph Neural Network for Interpretable Facial Weakness Assessment in Facioscapulohumeral Muscular Dystrophy.”**

HGNN combines facial action units and landmark-derived geometric measurements from eight standardized expressions to jointly perform:

- **Severity grading:** absence, moderate, or severe facial weakness.
- **Asymmetry prediction:** non-asymmetry, left-side predominance, or right-side predominance.

The repository includes the HGNN model, training and evaluation, machine-learning and deep-learning comparisons, ablation experiment, ROC analysis, Shaply interpretation, data distribution, landmark detection, and statistical test. See the [manuscript](paper.pdf) and [supplementary material](supplementary.pdf) for the study design and scientific context.

![Overview of the facial weakness assessment workflow](assets/figure_1.png)

## Method

Each participant contributes eight expressions, ordered as follows. The strings below are also the data keys used by [config.py](config.py):

1. `resting position`
2. `closing the eyes gently`
3. `closing the eyes firmly`
4. `raising the eyebrows`
5. `frowning`
6. `pursing the lips`
7. `showing the teeth`
8. `puffing of the cheeks`

Each expression contains 41 numerical features: 17 action-units (AU) and 24 facial-asymmetry (FA) measurements, comprising 10 bilateral deviations, 7 left-side measurements, and 7 right-side measurements. A participant is therefore represented by an `8 × 41` feature matrix.

HGNN uses three anatomically informed graphs:

| Graph                   | Nodes              | Information represented                                        |
|-------------------------|--------------------|----------------------------------------------------------------|
| Expression graph, `G_e` | 8 expressions      | Relationships between facial expressions                       |
| Action graph, `G_au`    | 17 AUs             | Relationships between facial actions                           |
| Asymmetry graph, `G_fa` | 24 FA measurements | Relationships between bilateral and left/right facial geometry |

The graph encoders contain three graph-convolution layers with widths `32 → 64 → 128`, each followed by global graph attention (GGA). Features from the three layers are concatenated and projected into 128-dimensional embeddings. Task-specific predictors combine the relevant embeddings to produce two sets of three-class logits. Graph connectivity is defined in [graph.py](graph.py), and the architecture is defined in [model.py](model.py).

## Repository

| File                                                           | Content                                                                                         |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| [config.py](config.py)                                         | Expression, feature, and class names and ordering                                               |
| [graph.py](graph.py)                                           | Expression, action, and facial-asymmetry graph connectivity                                     |
| [model.py](model.py)                                           | HGNN, graph convolution, GGA, projectors, predictors, and cross-constraint                      |
| [train_eval.py](train_eval.py)                                 | Main training and evaluation experiment                                                         |
| [compare_ML.py](compare_ML.py)                                 | XGBoost, Random Forest, SVM, and threshold-voting baseline experiments                          |
| [compare_model.py](compare_model.py)                           | CNN-2D, CNN-1D, MLP, and FT-Transformer model definitions                                       |
| [compare_DL.py](compare_DL.py)                                 | DL comparison experiments using models defined in [compare_model.py](compare_model.py)          |
| [ablation_model.py](ablation_model.py)                         | Ablation models defined by configurable graph, module, and model depth                          |
| [compare_ablation.py](compare_ablation.py)                     | Ablation experiments using models defined in [ablation_model.py](ablation_model.py)             |
| [metrics.py](metrics.py)                                       | Per-class metrics and overall multi-class accuracy                                              |
| [roc.py](roc.py)                                               | ROC analysis on two tasks                                                                       |
| [statistical_test.py](statistical_test.py)                     | Paired permutation tests against the main HGNN                                                  |
| [Shapley.py](Shapley.py)                                       | Task-level and class-specific SHAP interpretation                                               |
| [data_statistic.py](data_statistic.py)                         | Class-wise feature summaries                                                                    |
| [get_facial_landmarks.py](get_facial_landmarks.py)             | OpenFace, FAN, and Py-Feat landmark detection                                                   |
| [check_facial_landmarks.py](check_facial_landmarks.py)         | Inter-detector landmark comparisons                                                             |
| [facial_asymmetry_func.py](facial_asymmetry_func.py)           | Geometric facial measurement functions for asymmetrical features                                |
| [util.py](util.py)                                             | Training data scaling, fold preparation, learning-rate/early-stop function, and metric printing |
| [environment.yml](environment.yml)                             | Conda and Python dependencies                                                                   |
| [paper.pdf](paper.pdf), [supplementary.pdf](supplementary.pdf) | Paper and supplementary documents                                                               |

## Environment

The reported experiments used an NVIDIA RTX 3090 GPU and an Intel Xeon CPU. The supplied environment targets the Python 3.8 / PyTorch 1.9 stack with CUDA 11.1 wheels.

From the directory containing the scripts:

```bash
conda env create -f environment.yml
conda activate hgnn-fshd
```

Core pinned versions are listed below; [environment.yml](environment.yml) is the complete dependency specification.

| Component                | Version              |
|--------------------------|----------------------|
| Python                   | 3.8.18               |
| PyTorch / torchvision    | 1.9.0+cu111 / 0.10.0 |
| PyTorch Geometric        | 2.0.4                |
| NumPy / SciPy            | 1.23.5 / 1.10.1      |
| scikit-learn / XGBoost   | 1.3.2 / 1.7.6        |
| pandas / openpyxl        | 1.5.3 / 3.1.2        |
| OpenCV Python            | 4.8.1.78             |
| face-alignment / Py-Feat | 1.3.5 / 0.6.1        |
| OpenFace native toolkit  | 2.2.0                |

The training and evaluation scripts default to `cuda:0`. Use `--device cuda:0` to select the target device.

### OpenFace native dependency

Landmark processing requires the [OpenFace facial behavior analysis toolkit](https://github.com/TadasBaltrusaitis/OpenFace), release `OpenFace_2.2.0`, and its `FaceLandmarkImg` executable. Install its native prerequisites following the [official Unix installation guide](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation):

```bash
git clone --depth 1 --branch OpenFace_2.2.0 https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
bash download_models.sh
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
export PATH="$PWD/build/bin:$PATH"
```

Make the executable available in the shell running the scripts. Verify installation with a single-face image:

```bash
FaceLandmarkImg -f /path/to/test_image.png -out_dir /path/to/test_output -2Dfp
```

The resulting CSV should contain `confidence`, `x_0`–`x_67`, and `y_0`–`y_67`. FAN uses the pinned `face-alignment` interface with RGB input; its pretrained weights must be available locally or downloaded at initialization.

## Data format

`train_eval.py` expects a pickle file `AU_asymmetry_data.pkl` by default. It contains a dictionary with one original record per participant, keyed by a de-identified subject identifier:

```text
data[subject_id]
├── label: one-hot numerical array [2, 3]
├── resting position
│   ├── Action Unit: {AU key: numerical value}                 # 17 values
│   └── Facial Asymmetry
│       ├── Deviation: {deviation key: numerical value}        # 10 values
│       ├── Left: {unilateral key: numerical value}            # 7 values
│       └── Right: {unilateral key: numerical value}           # 7 values
└── ... same structure for the other seven expressions
```

Each label row has one active class:

| Task axis      | Class index 0   | Class index 1 | Class index 2 |
|----------------|-----------------|---------------|---------------|
| `0`: severity  | `absence`       | `moderate`    | `severe`      |
| `1`: asymmetry | `non-asymmetry` | `left`        | `right`       |

Feature names must match `ACTION_UNIT_KEYS`, `ASYMMETRY_DEVIATION_KEYS`, and `ASYMMETRY_SIDE_KEYS` in [config.py](config.py). Inputs are assembled in that explicit order, with `[17 AU, 10 deviation, 7 left, 7 right]` columns.

| Representation                      | Shape                                                   |
|-------------------------------------|---------------------------------------------------------|
| AU / FA arrays for `N` participants | `[N, 8, 17]` / `[N, 8, 24]`                             |
| HGNN and DL experiment input        | `[N × 8, 41]`                                           |
| ML classifier input                 | `[N, 328]`                                              |
| Labels and model output             | `[N, 2, 3]`; labels are one-hot, raw outputs are logits |

## Experiments

| Experiment codes      | Configurable arguments                                                                                   |
|-----------------------|----------------------------------------------------------------------------------------------------------|
| `train_eval.py`       | Data path, checkpoint directory, device, fold count, scaling, augmentation, and training hyperparameters |
| `compare_ML.py`       | Checkpoint directory, fold count, scaling, and `--models`                                                |
| `compare_DL.py`       | Fold/training options and `--models`                                                                     |
| `compare_ablation.py` | Fold/training options and `--variants`                                                                   |
| `roc.py`              | Checkpoint directory, device, fold count, and scaling                                                    |
| `statistical_test.py` | Fold options, `--variants`, and `--n-resamples`                                                          |
| `Shapley.py`          | `--data-path`, `--model-path`, and `--device`                                                            |
| `data_statistic.py`   | `--data-path` and `--output-path`                                                                        |

### Training and evaluation

Defaults are defined in [train_eval.py](train_eval.py):

| Argument                           | Default                      |
|------------------------------------|------------------------------|
| `--data-path`                      | `AU_asymmetry_data.pkl`      |
| `--checkpoint-dir`                 | `/data/FSHD/checkpoints/EXP` |
| `--k-fold`                         | 10                           |
| `--max-epoch`                      | 10,000                       |
| `--initial-lr`                     | `1e-4` (Adam)                |
| `--min-lr`                         | `1e-5`                       |
| `--patience-lr` / `--decay-factor` | 50 / `0.95`                  |
| `--patience-early-stop`            | 200                          |
| `--flip-aug`                       | Enabled                      |
| `--scaler`                         | Enabled                      |

Choose a checkpoint directory, then run:

```bash
python train_eval.py --data-path ./AU_asymmetry_data.pkl --checkpoint-dir ./checkpoints/EXP --device cuda:0
```

The main, ML, DL, and ablation experiments share [metrics.py](metrics.py) and the result printer in [util.py](util.py). Each fold reports both tasks:

- Overall multi-class accuracy;
- Per-class sensitivity, specificity, precision, and one-vs-rest accuracy.

### ML and DL comparisons

Select one or more comparison models to run:

```bash
python compare_ML.py --checkpoint-dir ./checkpoints/EXP --models xgb rf svm threshold
python compare_DL.py --checkpoint-dir ./checkpoints/EXP --models cnn2d ft_transformer
```

Both experiments reuse `data_1.pkl`–`data_10.pkl`. Keep the fold count, scaling setting, and training hyperparameters consistent with the main run. Supported comparison models:

| ML models                        | DL models                           |
|----------------------------------|-------------------------------------|
| XGBoost (`xgb`)                  | 2D CNN (`cnn2d`)                    |
| Random Forest (`rf`)             | 1D CNN (`cnn1d`)                    |
| Support Vector Machine (`svm`)   | Feed-forward neural network (`mlp`) |
| Threshold baseline (`threshold`) | FT-Transformer (`ft_transformer`)   |

### Ablation experiments

```bash
python compare_ablation.py --checkpoint-dir ./checkpoints/EXP --device cuda:0
```

Use `--variants` followed by one or more identifiers to run, for example `--variants without_gga depth_1`. Supported ablation models:

| Identifier                 | Ablation configuration                             |
|----------------------------|----------------------------------------------------|
| `ge`                       | Expression graph only                              |
| `ge_gau`                   | Expression and action graphs; no asymmetry graph   |
| `ge_gfa`                   | Expression and asymmetry graphs; no action graph   |
| `gau_gfa`                  | Action and asymmetry graphs; no expression graph   |
| `without_gga`              | Remove global graph attention                      |
| `without_cross_constraint` | Remove the cross-constraint                        |
| `depth_1`                  | One graph-convolution/GGA layer, width 32          |
| `depth_2`                  | Two graph-convolution/GGA layers, widths 32 and 64 |

The script trains the ablated variants using the same training protocol.

### ROC analysis

```bash
python roc.py --checkpoint-dir ./checkpoints/EXP --device cuda:0
```

This script computes two task-level ROCs and auc values. The two figure shows ten fold curves, their mean curve, and a ±1 SD band.

### Statistical tests

After the main and ablation experiments have completed:

```bash
python statistical_test.py --checkpoint-dir ./checkpoints/EXP --device cuda:0
```

The paired permutation test uses 100,000 resamples by default (`--n-resamples`).

### SHAP interpretation

[Shapley.py](Shapley.py) expects inputs `--data-path`, `--model-path`, and `--scalers-path`, run:

```bash
python Shapley.py --data-path ./AU_asymmetry_data.pkl --model-path ./checkpoints/EXP/model_1.pth --scalers-path ./checkpoints/EXP/scalers_1.pkl --device cuda:0
```
It reports expression and FA groups for both tasks, and AU features for severity. 

### Data distribution

```bash
python data_statistic.py --data-path ./AU_asymmetry_data.pkl --output-path ./data_statistic.xlsx
```

With `DATA_PATH` set to the prepared dataset, it writes `data_statistic.xlsx` containing class-wise AU, FA, and expression means for both tasks.

### Landmark detection

Set `INPUT_ROOT` and `RESULT_ROOT` in `get_facial_landmarks.py`, or `IMAGE_ROOT` in `check_facial_landmarks.py`, run:

```bash
python get_facial_landmarks.py
python check_facial_landmarks.py
```

The first script processes photographs with OpenFace, FAN, and Py-Feat. The second compares corresponding 68-point landmark sets across three detector pairs. 

## Citation

If you use this implementation, please cite the paper:

> Jiajing Zhang, Michael Kwan Leung Yu, Andong Wang, Jose Alberto Espino Pitti, Cheng Yin, Wei-Ning Lee, and Sophelia Hoi Shan Chan. *Hierarchical Graph Neural Network for Interpretable Facial Weakness Assessment in Facioscapulohumeral Muscular Dystrophy.*
