# HGHMDTA
# Source codes:
create_data.py: create data in pytorch format
utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
training.py: train the FMDTA model.
ginconv.py： model file.
# Requirements
	Python==3.9.21
	numpy==1.26
	pandas==2.2.3
	pytorch==2.1.2
	pytorch-cuda==11,8
	rdkit==2022.9.5  
# Dataset：
Our drug molecular graph data were obtained from GraphDTA（https://github.com/thinng/GraphDTA）
Our target structure graph data were obtained from DGraphDTA（https://github.com/595693085/DGraphDTA）

    python create_data.py
This returns davis_train.csv, and davis_test.csv, kiba_train.csv, kiba_test.csv.
# Operating Mode
Train a prediction model
	python training.py 0 0
