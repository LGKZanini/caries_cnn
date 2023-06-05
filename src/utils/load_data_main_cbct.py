import os
import numpy as np

path_load = './data/result.npy'

def make_paths(path, perm):
    
    labels_dict = np.load(path, allow_pickle=True)
    
    paths = {
        0 : [],
        1 : [],
        2 : [],
        3 : [],
        4 : []
    }
    
    for key in labels_dict.any().keys():
        
        labels = labels_dict.any().get(key)
        slices = len(labels_dict.any().get(key)[0])

        for s in range(slices):
            
            paths[labels[0][s]].append('./data/dente_'+key+'_slice_'+str(s)+'.npy')
        
            for p in range(perm):
                
                paths[labels[0][s]].append('./data/dente_'+key+'_slice_'+str(s)+'_permutation_'+str(p)+'.npy')
    
    return paths


def create_folds(path_data, fold):
    
    len_score = {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0,
    }
    
    for i in range(0,5):
    
        len_score[i] = len(path_data[i]) / fold
    
    folds = []
    
    for f in range(fold):
        
        data = []
        
        for i in range(0, 5):
            
            l = int(len_score[i]*f)
            r = int(len_score[i]*(f+1))
            
            for path in path_data[i][l:r]:
                
                data.append([path, i])            
        
        folds.append(data)
            
    return folds

def create_cross_val(folds, fold):
    
    data_train = []
    data_val = []
    
    for i, data in enumerate(folds):
        
        if i == fold:
            
            for insert in data:
                
                data_val.append(insert)
            
            continue
    
        for insert in data:

            data_train.append(insert)

    
    return data_train, data_val


def make_folds(total_folds, perm=64, ssl=False):
    
    if ssl:
        
        path_data = make_path_ssl()
        folds = create_folds(path_data, fold=total_folds)
        
    else:
    
        path_data = make_paths(path_load, perm=perm)
        folds = create_folds(path_data, fold=total_folds)
        
    return folds
    

def make_path_ssl():
    
    path_load_ssl = './data/ssl/'
    paths_result = []

    for dire in os.listdir(path_load_ssl):
        
        actual_path = path_load_ssl+dire

        paths = os.listdir(actual_path)
    
        for path in paths:
            
            paths_result.append(actual_path+'/'+path)
            
    return paths_result