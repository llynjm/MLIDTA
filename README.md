# MLI-DTA

MLI-DTA is a drug-target affinity prediction model that uses drug SMILES, drug molecular graphs, protein sequences, and protein contact graphs. The code in this repository provides data preprocessing, five-fold cross-validation splitting, model training, and evaluation.

## Source code

| File | Description |
| --- | --- |
| `create_data.py` | Converts Davis and KIBA data into PyTorch/PyTorch Geometric format and generates five-fold train/test splits. |
| `utils.py` | Provides the `TestbedDataset` class, mini-batch collation function, and evaluation metrics. |
| `models1.py` | Defines the MLI-DTA model architecture. |
| `training.py` | Trains and evaluates MLI-DTA under five-fold cross-validation. |

## Environment configuration and dependencies

The experiments were run on Linux with Python 3.9.21. PyTorch 2.1.2 was used as the deep learning framework, and graph neural network operations were implemented with PyTorch Geometric 2.6.1. RDKit was used for SMILES parsing and molecular graph construction. NumPy, Pandas, SciPy, scikit-learn, NetworkX, and Matplotlib were used for numerical computation, data processing, metrics, graph processing, and visualization. Model training was performed on a CUDA-enabled NVIDIA GPU with CUDA 11.8.

Main dependencies:

```text
python==3.9.21
numpy==1.26
pandas==2.2.3
pytorch==2.1.2
pytorch-cuda==11.8
rdkit==2022.9.5
torch==2.1.2
torch-geometric==2.6.1
scipy==1.13.1
scikit-learn==1.6.1
networkx==3.2.1
matplotlib==3.9.2
```

Example Conda environment setup:

```bash
conda create -n mlidta python=3.9.21 -y
conda activate mlidta

conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge rdkit==2022.9.5 numpy==1.26 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.6.1 networkx==3.2.1 matplotlib==3.9.2 -y
pip install torch-geometric==2.6.1
```

If your CUDA, PyTorch, or PyTorch Geometric versions differ, install the matching PyG packages according to the official PyTorch Geometric installation instructions.

## Dataset

The model uses Davis and KIBA benchmark datasets.

- Drug molecular graph data follow GraphDTA: https://github.com/thinng/GraphDTA
- Protein contact graph data follow DGraphDTA: https://github.com/595693085/DGraphDTA

Download the required dataset files according to the instructions in the two repositories above. After downloading, place the `data` directory in the same folder as the source code. The expected project structure is:

```text
MLI/
├── create_data.py
├── models1.py
├── training.py
├── utils.py
└── data/
    ├── davis/
    │   ├── ligands_can.txt
    │   ├── proteins.txt
    │   ├── Y
    │   ├── folds/
    │   ├── aln/
    │   └── pconsc4/
    └── kiba/
        ├── ligands_can.txt
        ├── proteins.txt
        ├── Y
        ├── folds/
        ├── aln/
        └── pconsc4/
```

## Five-fold data splitting

Run the following commands in the repository root:

```bash
cd MLI
```

Generate five folds for Davis:

```bash
python create_data.py --dataset davis --all-folds --n-folds 5 --seed 2024
```

Generate five folds for KIBA:

```bash
python create_data.py --dataset kiba --all-folds --n-folds 5 --seed 2024
```

These commands generate files such as:

```text
data/davis_fold0_train.csv
data/davis_fold0_test.csv
data/kiba_fold0_train.csv
data/kiba_fold0_test.csv
```

The same procedure is repeated for folds 0 to 4.

## Model training

Train all five folds on Davis:

```bash
python training.py --dataset davis --run-all-folds --n-folds 5 --seed 2024 --device cuda:0 --epochs 1000 --batch-size 512
```

Train all five folds on KIBA:

```bash
python training.py --dataset kiba --run-all-folds --n-folds 5 --seed 2024 --device cuda:0 --epochs 1000 --batch-size 512
```

Train a single fold, for example fold 0 on Davis:

```bash
python training.py --dataset davis --fold 0 --n-folds 5 --seed 2024 --device cuda:0 --epochs 1000 --batch-size 512
```

If no CUDA GPU is available, use:

```bash
python training.py --dataset davis --fold 0 --n-folds 5 --seed 2024 --device cpu --epochs 1000 --batch-size 512
```

## Output files

Training results are saved in:

```text
runs_5fold/
```

For each fold, the training script saves:

| Output file | Description |
| --- | --- |
| `model_GINConvNet_<dataset>_fold<k>_seed<seed>.pt` | Best model parameters selected by the lowest MSE on the test fold. |
| `result_GINConvNet_<dataset>_fold<k>_seed<seed>.csv` | Best RMSE, MSE, Pearson, Spearman, CI, Rm2, and best epoch for the fold. |
| `log_GINConvNet_<dataset>_fold<k>_seed<seed>.csv` | Epoch-level training loss and evaluation metrics. |
| `mse_curve_GINConvNet_<dataset>_fold<k>_seed<seed>.jpg` | MSE curve during training, generated when Matplotlib is installed. |
| `summary_GINConvNet_<dataset>_seed<seed>.csv` | Mean and standard deviation across the five folds. |

## Notes

- The default random seed is `2024`.
- The default number of folds is `5`.
- The default training epoch number is `1000`.
- The default batch size is `512`.
- The protein contact graphs are loaded from the `pconsc4` files in the dataset directory.
- The script reports RMSE, MSE, Pearson correlation, Spearman correlation, CI, and Rm2.

