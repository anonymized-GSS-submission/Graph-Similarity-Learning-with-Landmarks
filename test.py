from multiprocessing import context
import time
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader
# from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv,GCNConv
from torch_geometric.transforms import OneHotDegree
# from torch_geometric.data import Batch
# from torchinfo import summary
# from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import softmax, degree, sort_edge_index,to_dense_batch
import os
from torch.optim import lr_scheduler

import datetime
import prettytable as pt

#my import
from arg_parser import getargs
from GraphSim import graphsim
from EGSCS import EGSCStudent
from EGSCT import EGSCTeacher
from utils import *
from sgim import SGim
from simgnn import simgnn

def test(modelname):
    model = torch.load(save_file_name + 'minvaild_{}.pt'.format(k))
    prediction_mat = []
    groundtrue_mat = []
    # print(len(test_datasets))
    test_data_all = list(myproduct(train_datasets,test_datasets))
    dl = DataLoader(test_data_all,shuffle=False,batch_size=args.batch_size,num_workers = 10,pin_memory = True)
    loss = []
    model.eval()
    model.to(device)
    # device = next(model.parameters()).device
    with torch.no_grad():
        pbar = tqdm(dl) if args.verbose else dl
        for s,t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s,t)
            label = ged[s.i,t.i].to(device)
            prediction_mat.append(pred)
            groundtrue_mat.append(label)
            # idx[s.i,t.i - 600] = 1
            # idx.append([s.i.cpu().numpy().tolist(),t.i.cpu().numpy().tolist()])
            lossdata = F.mse_loss(pred,label,reduction = 'none').cpu().numpy()
            # loss.extend(lossdata)
            if args.verbose:
                pbar.set_description("test loss is {:.6f}".format(np.mean(lossdata)))

        prediction_mat = torch.cat(prediction_mat,dim = 0)
        groundtrue_mat = torch.cat(groundtrue_mat,dim = 0)

        p_mat = prediction_mat.reshape(len(train_datasets),len(test_datasets)).T.cpu().numpy()
        gt = groundtrue_mat.reshape(len(train_datasets),len(test_datasets)).T.cpu().numpy()
        rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(p_mat, gt,verbose = args.verbose)
        loss = F.mse_loss(prediction_mat,groundtrue_mat).cpu().numpy()

        tqdm.write("------------------------------------------------")
        tqdm.write("mse is {}  10^-3".format(loss))
        tqdm.write("rho is {}".format(np.mean(rho_list)))
        tqdm.write("tau is {}".format(np.mean(tau_list)))
        tqdm.write("p@10 is {}".format(np.mean(prec_at_10_list)))
        tqdm.write("p@20 is {}".format(np.mean(prec_at_20_list)))
    return loss*1000,np.mean(rho_list),np.mean(tau_list),np.mean(prec_at_10_list),np.mean(prec_at_20_list)


def makehtml(args):
    end_time = datetime.datetime.now()
    used_time = end_time - start_time
    keys = vars(args)
    html_meaasge = ''
    for key,value in keys.items():
        html_meaasge += str(key) + ' : ' + str(value) + '<br/>'
    html_meaasge += "use time = {} <br/>".format(str(used_time)) 
    print("used time is {} ".format(str(used_time)))
    return html_meaasge

def PrintInfo(dataset,args):
    print(f'==============  Dataset INFO: {dataset}  ===========================\n')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('\n===================  Args INFO  =================================\n')
    tb =  pt.PrettyTable()

    tb.field_names = ['Hyperparameter','values']
    keys = vars(args)
    for key,value in keys.items():
        tb.add_row([key,value])
    tb.align = 'l'
    print(tb)

if __name__ == "__main__":

    args = getargs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_time = datetime.datetime.now()
    start = time.time()

    datasetsname, lr = args.datasets,args.lr

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

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
    modelname = globals()[args.model]
    PrintInfo(train_datasets_all, args)

    mse_arr = []
    rho_arr = []
    tau_arr = []
    prec_at_10_arr = []
    prec_at_20_arr = []
    idx = int(len(train_datasets_all)*0.75)
    indices = np.random.permutation(len(train_datasets_all))
    train_datasets,vaild_datasets = train_datasets_all[indices[:idx].tolist()],train_datasets_all[indices[idx:].tolist()]
    
    save_file_name = "./result/{}/{}_{}/".format(args.name,datasetsname, args.model)
    for k in range(args.k_cross):
        tqdm.write("*****************k_corss : {}****************".format(k))

        # model = modelname(train_datasets_all.num_features,args).to(device)
        # optim = torch.optim.Adam(lr=lr, params=model.parameters())
        ged = torch.exp(-train_datasets.norm_ged).to(device)
        mse,rho,tau,prec_at_10,prec_at_20 = test(modelname)
        print(mse)
        print(rho)
        print(tau)
        print(prec_at_10)
        print(prec_at_20)