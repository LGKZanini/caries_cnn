import numpy as np  # pyright: ignore[reportMissingImports]
import wandb # pyright: ignore[reportMissingImports]

from sklearn.metrics import confusion_matrix # pyright: ignore[reportMissingImports]
    
def metrics_caries_icdas( y_val, y_pred, loss_item):
    

    print(y_val)
    print(y_pred)
    
    y_val = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    print(y_val)
    print(y_pred)
    
    conf = confusion_matrix(y_val, y_pred, labels=[0,1,2,3,4])
    
    FP = conf.sum(axis=0) - np.diag(conf)  
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    recall = TP/(TP+FN) 
    specificty = TN/(TN+FP) 
    precision = TP/(TP+FP) 
    acc = sum(np.diag(conf)) / conf.sum()

    wandb.log({
        "Loss_val" : sum(loss_item) / len(loss_item),
        "Accuracy" : acc,
        'Precision 0' : precision[0],
        'Precision 1' : precision[1],
        'Precision 2' : precision[2],
        'Precision 3' : precision[3],
        'Precision 4' : precision[4],
        'Specificity 0' : specificty[0],
        'Specificity 1' : specificty[1],
        'Specificity 2' : specificty[2],
        'Specificity 3' : specificty[3],
        'Specificity 4' : specificty[4],
        'Recall 0' : recall[0],
        'Recall 1' : recall[1],
        'Recall 2' : recall[2],
        'Recall 3' : recall[3],
        'Recall 4' : recall[4],
    })
    
    return

def metrics_caries_rotate( y_val, y_pred, loss_item):
    
    y_val = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    conf = confusion_matrix(y_val, y_pred, labels=[0,1,2,3])
    
    FP = conf.sum(axis=0) - np.diag(conf)  
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    recall = TP/(TP+FN) 
    specificty = TN/(TN+FP) 
    precision = TP/(TP+FP) 
    acc = sum(np.diag(conf)) / conf.sum()

    wandb.log({
        "Loss_val" : sum(loss_item) / len(loss_item),
        "Accuracy" : acc,
        'Precision 0' : precision[0],
        'Precision 90' : precision[1],
        'Precision 180' : precision[2],
        'Precision 270' : precision[3],
        'Specificity 0' : specificty[0],
        'Specificity 90' : specificty[1],
        'Specificity 180' : specificty[2],
        'Specificity 270' : specificty[3],
        'Recall 0' : recall[0],
        'Recall 90' : recall[1],
        'Recall 180' : recall[2],
        'Recall 270' : recall[3],
    })
    
    return


def metrics_caries_jigsaw( y_val, y_pred, loss_item):
    
    y_val = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    conf = confusion_matrix(y_val, y_pred, labels=[0,1,2,3,4,5,6,7,8])
    
    FP = conf.sum(axis=0) - np.diag(conf)  
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    recall = TP/(TP+FN) 
    specificty = TN/(TN+FP) 
    precision = TP/(TP+FP) 
    acc = sum(np.diag(conf)) / conf.sum()

    wandb.log({
        "Loss_val" : sum(loss_item) / len(loss_item),
        "Accuracy" : acc,
        
        'Precision 1' : precision[0],
        'Precision 2' : precision[1],
        'Precision 3' : precision[2],
        'Precision 4' : precision[3],
        'Precision 5' : precision[4],
        'Precision 6' : precision[5],
        'Precision 7' : precision[6],
        'Precision 8' : precision[7],
        'Precision 9' : precision[8],

        'Specificity 1' : specificty[0],
        'Specificity 2' : specificty[1],
        'Specificity 3' : specificty[2],
        'Specificity 4' : specificty[3],
        'Specificity 5' : specificty[4],
        'Specificity 6' : specificty[5],
        'Specificity 7' : specificty[6],
        'Specificity 8' : specificty[7],
        'Specificity 9' : specificty[8],

        'Recall 1' : recall[0],
        'Recall 2' : recall[1],
        'Recall 3' : recall[2],
        'Recall 4' : recall[3],
        'Recall 5' : recall[4],
        'Recall 6' : recall[5],
        'Recall 7' : recall[6],
        'Recall 8' : recall[7],
        'Recall 9' : recall[8],
    })
    
    return