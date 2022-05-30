#!/usr/bin/env python
# coding: utf-8

# # Fast Graph Similarity Task using Kernel Methods(landmark)
from tkinter import E
import torch
import os
import torch_geometric
import random
import numpy as np
import matplotlib.pylab as pl
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch,to_networkx
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import dataloader
import pprint
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd
from tqdm.notebook import tqdm
import argparse
import sys
import time

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type = str ,help="IMDBMulti,AIDS700nef,LINUX")
parser.add_argument("--t", type = float, default=0.25,help = "threshod")
parser.add_argument("--gpu", type = int,default=0,help = 'defined which gpu should be used')
# parser.add_argument("--pca",type=int,default=7)
parser.add_argument("--model",type=str,default='xgboost',help='et,knn,rf,catboost,xgboost,lr,br')
args = parser.parse_args()

os.chdir("../")
sys.path.append("./")

from train import *
from utils import *

# from net.GMN import GMN
# from net.SimGNN import *
# from net.fast_model import *
from sgim import *
# from net.GraphSim import GraphSim
# from net.SimPleMean import *
# from net.oa import OA

from sklearn.decomposition import *
from pycaret.regression  import *
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error as MSE
import xgboost
from xgboost import XGBRegressor

def GetKernel(datasets_1,datasets_2,model):
    prediction_mat = []
    groundtrue_mat = []

    device = next(model.parameters()).device

    pair = list(myproduct(datasets_1, datasets_2))
    dl = DataLoader(pair, shuffle=False, batch_size=5000, num_workers=10,pin_memory=True)

    pred = []
    model.eval()

    with torch.no_grad():
        pbar = tqdm(dl)
        for s, t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s, t)
            label = ged[s.i, t.i]
            prediction_mat.append(pred)
            groundtrue_mat.append(label)

    prediction_mat = torch.cat(prediction_mat,dim = 0)
    groundtrue_mat = torch.cat(groundtrue_mat,dim = 0)

    p_mat = prediction_mat.reshape(len(train_datasets),-1).T.cpu().numpy()
    gt = groundtrue_mat.reshape(len(train_datasets),-1).T.cpu().numpy() 
    p_mat = (p_mat*100000).round() / 100000
    
    return p_mat,gt


def similarity(gt,threshold,Id_select,node_id):
    node_id = [node_id]*len(Id_select)
    similarity_metric = gt[Id_select,node_id]
    return True if np.all(similarity_metric <= threshold) else False


def getLandmark(gt,threshold):
    assert gt.shape[0] == gt.shape[1]
    Id = gt.shape[0]
    Id = np.array(list(range(Id)))
    Id_select = [0]
    # np.random.seed(8)
    # np.random.shuffle(Id)
    for i in Id:
        if similarity(gt,threshold,Id_select,i):
            Id_select.append(i)
    return Id_select

def process_features(xi,xj):
    return np.concatenate([xi,xj],axis = 0).tolist()


def main():
    model = torch.load(f"./result/full/{datasets}_SGim/minvaild_0.pt").cuda(gpu)
    p_mat_train = torch.load(f"./landmark_new/temp/{datasets}_p_mat_train.pt")
    gt_train = torch.load(f"./landmark_new/temp/{datasets}_gt_train.pt")
    p_mat_val = torch.load(f"./landmark_new/temp/{datasets}_p_mat_val.pt")
    gt_val = torch.load(f"./landmark_new/temp/{datasets}_gt_val.pt")
    p_mat_test = torch.load(f"./landmark_new/temp/{datasets}_p_mat_test.pt")
    gt_test = torch.load(f"./landmark_new/temp/{datasets}_gt_test.pt")

    del model
    torch.cuda.empty_cache()
    thresholds = {
        'AIDS700nef':0.45,#0.43
        'LINUX':0.7,#0.7
        'IMDBMulti':0.12#0.05
    }

    # landmark_id = getLandmark(p_mat_train,thresholds[datasets])#np.load(f"./landmark/{datasets}_xgboost_landmarkid.json.npy")
    landmark_id = np.load(f"./landmark_new/xgboost_save/{datasets}_xgboost_landmarkid.npy")
    print(len(landmark_id))
    # np.save(f"./landmark_new/xgboost_save/{datasets}_xgboost_landmarkid.npy",landmark_id)
    landmark_features_train = p_mat_train[:,landmark_id]

    train_pair = []
    train_target = []
    for ids,i in enumerate(landmark_features_train):
        for idx,j in enumerate(landmark_features_train):
            train_pair.append(process_features(i,j)) 
            train_target.append(gt_train[ids,idx])
    train_pair = np.array(train_pair)
    train_df = pd.DataFrame(train_pair, columns = [str(i) for i in range(len(train_pair[0]))])
    train_df['target'] = train_target
    train_df = shuffle(train_df)
    train_df =train_df.drop_duplicates()

    landmark_features_val = p_mat_val[:,landmark_id]

    val_pair = []
    val_target = []
    for ids,i in enumerate(landmark_features_val):
        for idx,j in enumerate(landmark_features_train):
            val_pair.append(process_features(i,j))
            val_target.append(gt_val[ids,idx])
    val_pair = np.array(val_pair)
    val_df = pd.DataFrame(val_pair, columns = [str(i) for i in range(len(val_pair[0]))])
    val_df['target'] = val_target


    landmark_features_test = p_mat_test[:,landmark_id]

    test_pair = []
    test_target = []
    for ids,i in enumerate(landmark_features_test):
        for idx,j in enumerate(landmark_features_train):
            test_pair.append(process_features(i,j))
            test_target.append(gt_test[ids,idx])
    test_pair = np.array(test_pair)
    test_df = pd.DataFrame(test_pair, columns = [str(i) for i in range(len(test_pair[0]))])
    test_df['target'] = test_target

    dtrain = xgboost.DMatrix(train_df.values[:,:-1],train_df.values[:,-1])
    dval = xgboost.DMatrix(val_df.values[:,:-1],val_df.values[:,-1])
    dtest = xgboost.DMatrix(test_df.values[:,:-1],test_df.values[:,-1])
    evallist = [(dval, 'eval'),(dtest, 'test') ,(dtrain, 'train')]
    best_model = xgboost.Booster({"nthread":200},model_file=f"./landmark_new/xgboost_save/{datasets}_xgboost.json")
    # best_model =xgboost.train(param1, dtrain, num_round,evallist)#,eval_metric =evallist
    dtest = xgboost.DMatrix(test_df.values[:,:-1])
    times = []
    for _ in range(10):
        start = time.time()
        best_model.predict(dtest)
        times.append(time.time() - start)
    print(MSE(best_model.predict(dtest),test_target)*1000)#flag," : ",
    print("Time: ", np.mean(times),np.std(times))
    print(times)
    pred_temp = best_model.predict(dtest)
    # pkl_filename = f"./landmark_new/xgboost_save/{datasets}_{args.model}.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(best_model, file)
    # best_model.save_model(f"./landmark_new/xgboost_save/{datasets}_{args.model}.json")
    pred_temp = np.array(pred_temp).reshape(len(test_datasets),-1)
    test_target = np.array(test_target).reshape(len(test_datasets),-1)
    rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(pred_temp, test_target,verbose = True)
    rho,tau,prec_at_10,prec_at_20 = np.mean(rho_list), np.mean(tau_list), np.mean(prec_at_10_list), np.mean(prec_at_20_list)
    mse = ((pred_temp - test_target)**2).mean()

    return mse,rho,tau,prec_at_10,prec_at_20



if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    datasets = args.datasets
    datasetsname = datasets
    model_name = args.model
    gpu = args.gpu
    mse_arr = []
    rho_arr = []
    tau_arr = []
    prec_at_10_arr = []
    prec_at_20_arr = []

    train_datasets_all = GEDDataset("../SomeData4kaggle/{}".format(datasetsname), name=datasetsname)
    test_datasets = GEDDataset("../SomeData4kaggle/{}".format(datasetsname), name=datasetsname, train=False)
    max_degree = 0
    for g in train_datasets_all + test_datasets:
        if g.edge_index.size(1) > 0:
            max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
    one_hot_degree = OneHotDegree(max_degree, cat=True)
    train_datasets_all.transform = one_hot_degree
    test_datasets.transform = one_hot_degree
    train_datasets_all = train_datasets_all.shuffle()
    idx = int(len(train_datasets_all)*0.75)
    indices = np.random.permutation(len(train_datasets_all))
    train_datasets,vaild_datasets = train_datasets_all[indices[:idx].tolist()],train_datasets_all[indices[idx:].tolist()]
    
    for i in range(20):
        print(vaild_datasets[i].i)
    ged = train_datasets.norm_ged
    ged = torch.exp(-ged)
    mse,rho,tau,prec_at_10,prec_at_20 = main()
    
    mse_arr.append(mse)
    rho_arr.append(rho)
    tau_arr.append(tau)
    prec_at_10_arr.append(prec_at_10)
    prec_at_20_arr.append(prec_at_20)

    print(mse*1000,'x 10^-3')
    print(rho)
    print(tau)
    print(prec_at_10)
    print(prec_at_20)

    result_DF = pd.DataFrame()
    result_DF['mse'] = mse_arr
    result_DF['rho'] = rho_arr
    result_DF['tau'] = tau_arr
    result_DF['p10'] = prec_at_10_arr
    result_DF['p20'] = prec_at_20_arr
