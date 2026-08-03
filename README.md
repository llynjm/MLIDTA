# MLIDTA
# Source codes:
create_data.py: create data in pytorch format

utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.

training.py: train the FMDTA model.

models1： model file.
# Environment Configuration and Version Dependencies
The experimental runtime environment is as follows: the operating system is Linux; the programming language is Python==3.9.21; the deep learning framework is PyTorch 2.1.2; graph neural network operations are implemented using PyTorch Geometric 2.6.1; RDKit is used for molecular graph construction and SMILES parsing; and NumPy, Pandas, and SciPy are used for numerical computations and data processing, respectively. Model training is performed on a CUDA-enabled NVIDIA GPU using CUDA version 11.8. The main dependencies are as follows:
	Python==3.9.21<br>
	numpy==1.26
	pandas==2.2.3
	pytorch==2.1.2
	pytorch-cuda==11,8
	rdkit==2022.9.5  
	torch==2.1.2
	torch-geometric==2.6.1
	scipy==1.13.1
	scikit-learn==1.6.1
	networkx==3.2.1
	matplotlib==3.9.2
# Dataset：
Our drug molecular graph data were obtained from GraphDTA（https://github.com/thinng/GraphDTA ）
Our target structure graph data were obtained from DGraphDTA（https://github.com/595693085/DGraphDTA ）
Simply download the dataset as described at the two links.

   python .\create_data.py --dataset davis --all-folds --n-folds 5 --seed 2024
   python .\create_data.py --dataset kiba --all-folds --n-folds 5 --seed 2024
The code above will split the dataset using 5-fold cross-validation.
# Operating Mode
After downloading the dataset file, place the dataset and the code files from this repository in the same folder. Run the code to start training the model; the model results will be saved.
	python .\training.py --dataset davis --run-all-folds --n-folds 5 --seed 2024 --device cuda:0 --epochs 1000 --batch-size 512
	python .\training.py --dataset kiba --run-all-folds --n-folds 5 --seed 2024 --device cuda:0 --epochs 1000 --batch-size 512
