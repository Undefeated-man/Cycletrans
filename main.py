import time
import dill
import ast
import os
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

from tqdm import tqdm, trange
from torch.cuda import amp
from collections import OrderedDict
from transformers import get_cosine_schedule_with_warmup
from utils import *
from model import *

import warnings
warnings.filterwarnings("ignore")

USE_CUDA = torch.cuda.is_available()

# Set random seed
seed = 7
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)

# initialize
set_seed(seed)
folder   = "/kaggle/input/csdt-train"
df       = [pd.read_csv(folder + '/%s.csv'%i) for i in ["train_pro", "test_pro", "valid_pro"]]

a        = pd.concat([ddf["ICD9_CODE"] for ddf in df]).reset_index(drop=True)
a        = a.apply(lambda x: ast.literal_eval(x))
syms_max = max(a.apply(lambda x: len(x)))

a        = pd.concat([ddf["PRO_CODE"] for ddf in df]).reset_index(drop=True)
a        = a.apply(lambda x: ast.literal_eval(x))
diag_max = max(a.apply(lambda x: len(x)))

a        = pd.concat([ddf["NDC"] for ddf in df]).reset_index(drop=True)
a        = a.apply(lambda x: ast.literal_eval(x))
med_max  = max(a.apply(lambda x: len(x)))


# Load dataset
train_df   = pd.read_csv(folder + '/train_pro.csv')
test_df    = pd.read_csv(folder + '/test_pro.csv')
valid_df   = pd.read_csv(folder + '/valid_pro.csv')

train_df["ICD9_CODE"]  = train_df["ICD9_CODE"] .apply(lambda x: ast.literal_eval(x))
train_df["PRO_CODE"] = train_df["PRO_CODE"].apply(lambda x: ast.literal_eval(x))
train_df["NDC"]  = train_df["NDC"] .apply(lambda x: ast.literal_eval(x))
test_df["ICD9_CODE"]   = test_df["ICD9_CODE"]  .apply(lambda x: ast.literal_eval(x))
test_df["PRO_CODE"]  = test_df["PRO_CODE"] .apply(lambda x: ast.literal_eval(x))
test_df["NDC"]   = test_df["NDC"]  .apply(lambda x: ast.literal_eval(x))
valid_df["ICD9_CODE"]  = valid_df["ICD9_CODE"] .apply(lambda x: ast.literal_eval(x))
valid_df["PRO_CODE"] = valid_df["PRO_CODE"].apply(lambda x: ast.literal_eval(x))
valid_df["NDC"]  = valid_df["NDC"] .apply(lambda x: ast.literal_eval(x))

with open(folder + "/sub2diag_dict_pro.pkl", "rb") as f:
    sub2diag_dict = dill.load(f)

with open(folder + "/voc_NDC_DRUG.pkl", "rb") as f:
    drug_namge_dict = dill.load(f)
    
with open(folder + "/voc_final.pkl", "rb") as f:
    voc   = dill.load(f)

with open(folder + "/ddi_A_final.pkl", "rb") as f:
    ddi_A = dill.load(f)
    

DRUG_NUM = len(voc["med_voc"].idx2word)
SYM_NUM  = len(voc["diag_voc"].idx2word)
HIDDEN   = 178

def train(net, cycle_net, criterion, train_set, valid_set,
          ddi_matrix, batch_size = 512, epochs = 10, 
          lr = 1e-3, max_grad_norm = 1000, 
          optimizer = None, scheduler = None, 
          cpu = False, params_pth = None, log_path = "./log/",
          gradient_accumulate_step = 1, output_loss = None):
    
    if not cpu:
        device_ids      = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]
    else:
        device_ids      = [torch.device("cpu")]
    if params_pth:
        print("Read parameters to finetune:")
        dic = torch.load(params_pth)
        net.load_state_dict(dic["csdt"]["model_state_dict"])
        cycle_net.load_state_dict(dic["drug2diag"]["model_state_dict"])

    print("\ntrain on %s\n"%str(device_ids))
    enable_amp  = True if "cuda" in device_ids[0].type else False
    scaler      = amp.GradScaler(enabled = enable_amp)
    net.to(device_ids[0])
    cycle_net.to(device_ids[0])
    train_iter  = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
#     train_iter  = zip(train_set.sub_train, train_set.hadm_train, train_set.sym_train, train_set.drug_train, train_set.diag_train)
    if not output_loss is None:
        loss_ls = output_loss
    else:
        loss_ls     = []
        
    prec = -np.inf
    
    for epoch in trange(epochs):
        if epoch % 2 == 0:
            eval_res = test(net, valid_set, ddi_matrix, batch_size = batch_size, \
             max_grad_norm = 1000, cpu = False, \
             log_path = "./log/", gradient_accumulate_step = gradient_accumulate_step)
            if prec < eval_res[-1]:
                prec = eval_res[-1]
                torch.save(
                        {
                            "csdt":{
                                'epoch': epoch,
                                'model_state_dict': net.state_dict()# ,
                            }, 
                            "drug2diag":{
                                'epoch': epoch,
                                'model_state_dict': cycle_net.state_dict(),
                            }
                        }, "CycleSafeDrugTransformer.params"
                )
            
        net.train()
        cycle_net.train()
        print("\n In epoch ", epoch)
        for idx, value in tqdm(enumerate(train_iter)):
            ini_time    = time.time()
            sub_ids, hamd_ids, sym, med = value
#             try:
            sym, med    = torch.tensor(padding_data(sym, SYM_NUM, fillin = SYM_NUM, mode = 0)), torch.tensor(med)
#             except Exception as e:
#                 print(e)
#                 print(sym)
#                 return

            net.syms_embd.to(torch.device("cpu"))
            cycle_net.syms_embd.to(torch.device("cpu"))
            cycle_net.drug_embd.to(torch.device("cpu"))

#             with amp.autocast(enabled = enable_amp):
            output, att    = net(sym, sub_ids, hamd_ids)
            output         = torch.sigmoid(output)
            output         = torch.where(output == 0, torch.tensor(1.0).to(device_ids[0]), output)
            res            = []
            idx_           = (output > .5).cpu()
            for i in range(len(output)):
                res.append(net.drug_embd.weight.data.cpu()[:-1, :][idx_[i]])
            out,drug_m     = pad(res, HIDDEN, True)
            drug_m         = torch.cat([torch.ones((len(drug_m), 1, 1)), torch.stack([m[:-1] for m in drug_m], dim = 0)], dim = 1)
            drug_m         = torch.matmul(drug_m, drug_m.transpose(-1, -2))
            out            = torch.stack(out, dim = 0)
            cycle      = cycle_net(out, drug_m, sym)
            cycle      = torch.sigmoid(cycle)
            cycle      = torch.where(cycle == 0, torch.tensor(1.0).to(device_ids[0]), cycle)
#             if idx % 2 == 0:
#                 cycle      = cycle_net(out, drug_m, sym)
#                 cycle      = torch.sigmoid(cycle)
#                 cycle      = torch.where(cycle == 0, torch.tensor(1.0).to(device_ids[0]), cycle)
#             else:
#                 with torch.no_grad():
#                     cycle  = cycle_net(out, drug_m, sym)
#                 cycle      = torch.sigmoid(cycle)
#                 cycle      = torch.where(cycle == 0, torch.tensor(1.0).to(device_ids[0]), cycle)
#             cycle          = None
            loss           = criterion(sym, cycle, med, output, ddi_matrix, att, device_ids[0])
            with torch.no_grad():
                loss_ls.append(float(loss.cpu().detach().numpy()))

#             scaler.scale(loss).mean().backward()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            if (idx + 1) % gradient_accumulate_step == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
                optimizer.step()
#                 scheduler.step()
                optimizer.zero_grad()
                net.zero_grad()
                if cycle_net:
                    cycle_net.zero_grad()

            if idx % 50  == 0:
                with torch.no_grad():
                    with open("loss.pkl", "wb") as f:
                        dill.dump(loss_ls, f)

            if not os.path.exists("./log"):
                os.mkdir("./log")

            with open(log_path + "log_train", "a") as f:
                f.write("Epoch %s, Batch %s: %.4f sec\n"%(epoch, idx, time.time() - ini_time))
            
    eval_res = test(net, valid_set, ddi_A, batch_size = batch_size, \
    max_grad_norm = 1000, cpu = False, \
    log_path = "./log/", gradient_accumulate_step = gradient_accumulate_step)
    if prec < eval_res[-1]:
        torch.save(
                {
                    "csdt":{
                        'epoch': epoch,
                        'model_state_dict': net.state_dict()# ,
                    }, 
                    "drug2diag":{
                        'epoch': epoch,
                        'model_state_dict': cycle_net.state_dict(),
                    }
                }, "CycleSafeDrugTransformer.params"
        )
        
    return loss_ls


def test(net, test_set,
          ddi_matrix, batch_size = 512, 
          max_grad_norm = 1000,  
          cpu = False, params_pth = None, log_path = "./log/",
          gradient_accumulate_step = 1):
    
    if not cpu:
        device_ids      = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]
    else:
        device_ids      = [torch.device("cpu")]
    if params_pth:
        print("Read parameters to test:")
        dic = torch.load(params_pth)
        net.load_state_dict(dic["csdt"]["model_state_dict"])

    print("\ntest on %s\n"%str(device_ids))
    enable_amp  = True if "cuda" in device_ids[0].type else False
    scaler      = amp.GradScaler(enabled = enable_amp)
    net.to(device_ids[0])
    train_iter  = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    result      = []
    target      = []
    ori_result  = []
    net.eval()
    print("\n Testing: ")
    
    for idx, value in enumerate(train_iter):
        ini_time    = time.time()
        sub_ids, hamd_ids, sym, med = value

        net.syms_embd.to(torch.device("cpu"))

        with torch.no_grad():
            output, _  = net(sym, sub_ids, hamd_ids)
            output     = torch.sigmoid(output)
            output     = torch.where(output == 0, torch.tensor(1.0).to(device_ids[0]), output)
        result.append(output.cpu().detach().numpy().tolist()[::])
        ori_result.append(_.cpu().detach().numpy())
        target.append(med.cpu().detach().numpy().tolist()[::])
        
    ori_res = result
    result  = [onehot2ids(r) for r in result]
    fin_res = []
    for r in result:
        for r_ in r:
            fin_res.append(r_)
    result  = fin_res
    
    fin_res = []
    for r in target:
        for r_ in r:
            fin_res.append(r_[:r_[-1]])
    target = fin_res
    with open("prediction.pkl", "wb") as f:
        dill.dump(result, f)

    with open("ground_truth.pkl", "wb") as f:
        dill.dump(target, f)
    
    with open("att_mat.pkl", "wb") as f:
        ori_result = np.concatenate(ori_result)
        np.save('att_mat.npy', ori_result)
    
#     print(len(result))
#     print(result)
    print("Precision:", prec := get_precision(result, target))
    print("Recall:", get_recall(result, target))
    print("F1-score:", get_F1_score(result, target))
    print("DDI rate:", np.mean([ddi_rate_calculation(ddi_A, result[i]) for i in range(len(result))]))
    print("Jaccard:", np.mean([jaccard_score_manual(i, j) for i, j in zip(target, result)]))
    print("Avg drug #:", np.mean([len(r) for r in result]))
    
    return result, target, ori_result, prec

if __name__ == "__main__":
    prec_s     = 0
    best_seed  = 51
    n_blocks   = 2
    ddi_mat    = ddi_A
    HIDDEN_DIM = 128
    batch_size               = 12
    epochs                   = [20, 30]
    lr                       = [5e-4, 1e-6]
    gradient_accumulate_step = 1
    lambda_weight            = [.1, (.3, 0), .2, 0, 0.02, 0, 0]

    train_set  = MyDataLoader(train_df)
    valid_set  = MyDataLoader(valid_df)
    test_set  = MyDataLoader(test_df)
    
    set_seed(best_seed)
    net        = CycleSafeDrugTransformer(None, ddi_mat = ddi_mat, sub2diag_dict = sub2diag_dict, n_blocks = n_blocks, cpu = False, dropout = .1, d_model = HIDDEN_DIM)
    cycle_net  = Drug2Diagnosis(net, n_blocks = 1, cpu = False, hidden_dim=HIDDEN_DIM)
    criterion  = CustomizedLoss(lambda_weight = lambda_weight, diag_num = SYM_NUM, drug_num = DRUG_NUM, model = net)
    optimizer  = [torch.optim.RAdam(net.parameters(), lr = lr[0])
    #               torch.optim.RAdam(net.parameters(), lr = lr[1])
                ]
    scheduler  = [ 
                get_cosine_schedule_with_warmup(optimizer = optimizer[0], num_warmup_steps = 0, 
                                            num_training_steps= len(
                                                    torch.utils.data.DataLoader(train_set, batch_size = batch_size)
                                                    ), 
                                            num_cycles = 0.5), 
    #               get_cosine_schedule_with_warmup(optimizer = optimizer[1], num_warmup_steps = 0, 
    #                                         num_training_steps= len(
    #                                                 torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    #                                                 ), 
    #                                         num_cycles = 0.5)
                ]
    
    ls = []
    # # Stage 0 training
    ls = train(net, cycle_net, criterion, train_set, valid_set, ddi_A, batch_size = batch_size, 
            epochs = epochs[0], lr = lr[0], max_grad_norm = 1000, optimizer = optimizer[0], 
            scheduler = scheduler[0], 
            cpu = False, params_pth = None, 
            log_path = "./log/", gradient_accumulate_step = gradient_accumulate_step, output_loss = ls)
    plt.plot(ls)
    
    # Evaluate
    result, target, ori, prec = test(net, test_set, ddi_mat, batch_size = batch_size, 
    max_grad_norm = 1000, cpu = False, 
    params_pth = "/kaggle/working/CycleSafeDrugTransformer.params",
    log_path = "./log/", gradient_accumulate_step = gradient_accumulate_step)