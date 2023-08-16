import numpy as np
import json
import os

def get_Path(rootDir):
    all_path = []
    files = os.listdir(rootDir)
    for file in files:
        path = rootDir+file
        all_path.append(path)

    return all_path


def splitdata(dataset,fold,index):
    length = len(dataset)
    data_indice = np.arange(length)
    data_indices = np.random.default_rng(42).permutation(data_indice)
    fold_length =int( length / fold)

    if index == 1:
        val_idx = data_indices[:fold_length]
        test_idx = data_indices[fold_length*(fold-1):]
        train_idx = data_indices[fold_length:fold_length*(fold-1)]
    elif index == fold:
        val_idx = data_indices[fold_length*(fold-1):]
        test_idx = data_indices[fold_length*(fold-2):fold_length*(fold-1)]
        train_idx = data_indices[:fold_length*(fold-2)]
    else:
        val_idx = data_indices[fold_length*(index-1):fold_length*index]
        test_idx = data_indices[fold_length*(index-2):fold_length*(index-1)]
        train_idx = np.concatenate([data_indices[:fold_length*(index-2)],
                                data_indices[fold_length*index:]])
        
    return train_idx,val_idx,test_idx




