import torch
import os
import argparse
import sys
import json
import math
from torch_geometric.loader import DataLoader
from datasets.SuperCon import SC
from features.process_data import splitdata,get_Path
from features.identity_disorder import classify

parser = argparse.ArgumentParser('Predicting value..', add_help=False)
parser.add_argument("--data_path", type=str, default='datasets/SuperCon')
parser.add_argument('--model', type=str, default='best_models/')
parser.add_argument('--feature_type', type=str, default='crystalnet')
parser.add_argument('--order_type', type=str, default='all')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pred_list= []

def get_prediction(model_path,test):

    results = []
    all_data = []

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    
    
    test_data = SC(args.data_path,'test', test,fold_id, feature_type=args.feature_type)

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(batch=data.batch,
                         edge_occu=data.edge_occu,
                         f_in=data.x,edge_src=data.edge_src,edge_dst=data.edge_dst,
                         edge_vec=data.edge_vec,edge_attr=data.edge_attr,edge_num=data.edge_num) 
            pred = pred.squeeze()
            
            pred = pred.detach() * model.task_std + model.task_mean
            pred = math.exp(pred)    
            info = {}
            info["id"] = str(data.name[0])
            info["pred"] = pred
            info['target'] = math.exp(float(data.y))
            results.append(info)
    return results

def get_one_prediction(model_path,x,batch,
                       edge_occu,edge_src,
                       edge_dst,edge_vec,
                       edge_attr,edge_num):


    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(batch=batch,edge_occu=edge_occu,
                     f_in=x,edge_src=edge_src,edge_dst=edge_dst,
                     edge_vec=edge_vec,edge_attr=edge_attr,edge_num=edge_num)
        pred = pred.squeeze()

        pred = pred.detach() * model.task_std + model.task_mean
        pred = math.exp(pred)

    return pred

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    data_source = get_Path(args.data_path+'/cif/')


    if args.order_type == 'order':
        order_data,disorder_data = classify(data_source)
        data_source = order_data
    elif args.order_type == 'disorder':
        order_data,disorder_data = classify(data_source)
        data_source = disorder_data
    elif args.order_type == 'all':
        data_source = data_source
    else:
        print('please input the currect order_type')
    
    
    fold_num = 10
    fold_id = 1
    all_model = get_Path(args.model)
    test_list = [] 
    for fold_idx in range(1,fold_num+1):
        train_i,valid_i,test_i = splitdata(data_source,fold_num,fold_idx)

        test_data = [data_source[i] for i in test_i]
        test_list.append(test_data)

    for model,test in zip(all_model,test_list):
        pred = get_prediction(model,test)
        pred_list.append(pred)
        fold_id += 1

    with open('pred.json','w') as f:
        json.dump(pred_list,f)
    print('Done')
        

