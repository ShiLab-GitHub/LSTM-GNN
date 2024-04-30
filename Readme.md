# LSTM_GCN

This reposiry contains the data and python scripts in support of the manuscript: LSTM-GNN: A multi-channel model for molecular proprieties prediction. The focus of this work is to provide a robust and reliable approach for researchers and practitioners in the field of cheminformatics and drug discovery.

# **Environment**

Run`pip install -r requirements.txt`

# **Usage**

For HIV dataset
```sh
python train.py --data_path data/MoleculeNet/hiv.csv --save_path model_save --log_path log --seed <seed> --epochs <epochs> --num_folds <folds>
```

---------

For MUV dataset
```sh
python train.py --data_path data/MoleculeNet/muv.csv --save_path model_save --log_path log --seed <seed> --epochs <epochs> --num_folds <folds>
```