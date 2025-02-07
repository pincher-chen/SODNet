# Learning Superconductivity from Ordered and Disordered Material Structures (NeuIPS 2024)

A transformer-based GNN model for learning superconductivity from ordered and disordered crystal structures. Implementation codes for Learning Superconductivity from Ordered and Disordered Material Structure.  
[**[Paper]**](https://neurips.cc/virtual/2024/poster/97553)

## Requirement

The important packages are presented as follows:

```
e3nn                  0.4.4
numpy                 1.22.4
pymatgen              2023.2.28
scipy                 1.8.1
timm                  0.4.12
torch                 1.10.2+cu111
torch-cluster         1.6.0
torch-geometric       2.2.0
torch-scatter         2.0.9
torch-sparse          0.6.13
torch-spline-conv     1.2.1
torchaudio            0.10.2 
torchmetrics          0.8.2
torchvision           0.11.3+cu111
tqdm                  4.65.0 
```

## Dataset

The dataset is undered `datasets/SuperCon/cif/` and the Tc values are saved in `datasets/SuperCon/df_all_data1202.csv`. More details can be found in datasets/SuperCon/README.

## Example

Some tests on data processing, modeling and inference are given in the `examples/test.py`
You can run the test with the following command and determine if your environment is installed correctly:

```
    python test.py
```

## Training

All the training scripts are under `scripts/SuperCon/` . 
The input data will be divided into 10-fold before training, so you can train according to the number of folds you want to run.
For example:

```
    sh scripts/SuperCon/train_[FOLD].sh
```
If you want to run all the folds at once, you can use the following command:

```
    sh scripts/SuperCon/train_all.sh
```

## Inference

After training, all models will be saved in `best_models/`.
You can use these `*_save.pt` files for inference with the following commands:

```
    sh scripts/infer/infer.sh
```
The results of inference will be saved in `pred.json`.

## Citation
Please consider citing our work if you find it helpful:

```
@inproceedings{chenlearning,
  title={Learning Superconductivity from Ordered and Disordered Material Structures},
  author={Chen, Pin and Peng, Luoxuan and Jiao, Rui and Mo, Qing and Zhen, WANG and Huang, Wenbing and Liu, Yang and Lu, Yutong},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```
