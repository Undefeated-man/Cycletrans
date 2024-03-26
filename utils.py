import torch.nn as nn
import dill

folder   = "/kaggle/input/csdt-train"
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

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

class Embedder(torch.nn.Module):
    def __init__(self, max_len = 64, d_model = 64):
        super(Embedder, self).__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.max_len = max_len
    
    def position_encoding(self):
        pos_embd = torch.arange(self.max_len, dtype=torch.float32).reshape(-1, 1) / \
                    torch.pow(10000, torch.arange(0, self.d_model, 
                                                  2, dtype=torch.float32) / self.d_model)
        embd     = torch.zeros((1, self.max_len, self.d_model))
        embd[:, :, 0::2] = torch.sin(pos_embd)
        embd[:, :, 1::2] = torch.cos(pos_embd)
        return embd
        
    def forward(self, x):
        return x + self.position_encoding().to(self.device)
    

class OneHotEncoder:
    """
        Args:
            - ls: a batch of ids, each ids is a sub-list of id
        
        Returns:
            To return the one-hot encoding result.
    """
    def __init__(self, num_class):
        self.num = num_class

    def __call__(self, ls):
        res = torch.zeros((len(ls), self.num + 1))
        for l in range(len(ls)):
            res[l, ls[l]] = 1
        return res[:, :-1]


class CustomizedLoss:
    """
        The customized loss function for the CycleSafeDrugTransformer model.

        Args:
            - lambda_weight: the weight to adjust each loss function importance. 
                Default: [.1, (1, 2), .7, .05, .01, 1] --> (cycle_loss, (drug_loss<classification>, drug_loss<distance>), 
                                                        ddi_loss, pred_drugs_type_loss, att_score_loss, len_loss)
    """
    def __init__(self, lambda_weight = [.1, (1, 2), .7, .05, .001, 1, 1], diag_num = SYM_NUM, drug_num = DRUG_NUM, model = None):
        self.classification_loss = nn.BCELoss()
        self.lambda_weight       = lambda_weight
        self.sigmoid             = nn.Sigmoid()
        self.diag_encoder        = OneHotEncoder(diag_num)
        self.drug_encoder        = OneHotEncoder(drug_num)
        self.model               = model

    def cls_loss(self, pred_drug, real_drug):
        tmp = torch.where(pred_drug == 0, torch.tensor(1.0).to(self.device), pred_drug)
        return self.classification_loss(pred_drug.float(), real_drug.float()) #- \
                #torch.mean(pred_drug * (torch.log(tmp) - 1))
    
    def cycle_loss(self, real_data, cycled_data):
        return torch.mean(torch.abs(real_data - cycled_data))

    def interact(self, drug_prob, diag_prob, ddi_matrix):
        """
            The interaction loss function between the drug and diagnosis.
            Args:
                - drug_prob: the drug predicted prob.
                - diag_prob: the diagnosis predicted prob.
                - ddi_matrix: the drug-drug interaction matrix.
        """
        new = True
        if new:
            loss          = torch.mean(torch.matmul(drug_prob, ddi_matrix))
        else:
            max_diag      = None # 1001
            drug_embd     = sig2embd(drug_prob, self.model.drug_embd.weight.data.cpu(), ddi_matrix.shape[0]).to(self.device)
            diag_embd     = sig2embd(diag_prob, self.model.diag_embd.weight.data.cpu(), max_diag).to(self.device)
            if max_diag:
                diag_embd = torch.mean(diag_embd, dim = 1).unsqueeze(0)
                inter     = self.sigmoid(torch.matmul(drug_embd, diag_embd.transpose(1, 2)))
            else:
                inter     = self.sigmoid(torch.matmul(drug_embd, diag_embd.T))
            loss          = torch.mean(torch.matmul(inter.transpose(-1, -2), ddi_matrix))
        return loss
    
    def earth_movers_distance_loss(self, pred, target):
        C      = pred.shape[-1]
        pred   = pred.view(-1, C)
        target = target.view(-1, C)
        pred_cumsum   = torch.cumsum(pred, dim = 1)
        target_cumsum = torch.cumsum(target, dim = 1)
        loss   = torch.sum(torch.abs(pred_cumsum - target_cumsum), dim = 1)
        return torch.mean(loss)
    
    def __call__(self, real_diag, cycled_diag, real_drug, pred_drug, ddi_matrix, att_score = None, device = torch.device("cuda:0")):
        real_diag   = self.diag_encoder(real_diag.cpu()).to(device)
        real_drug   = self.drug_encoder(real_drug.cpu()).to(device)
        real_diag   = real_diag.to(device)
        real_drug   = real_drug.to(device)
        ddi_matrix  = torch.tensor(ddi_matrix, dtype = torch.float32).to(device)
        self.device = device
        
        if self.lambda_weight[0]:
            cycle_loss  = self.lambda_weight[0] * self.cycle_loss(real_diag, cycled_diag)
        else:
            cycle_loss  = 0
            
        if self.lambda_weight[1] != (0, 0):
            drug_loss   = self.lambda_weight[1][0] * self.cls_loss(pred_drug.float(), real_drug) + \
                            self.lambda_weight[1][1] * self.cycle_loss(pred_drug, real_drug)
#             print(drug_loss)
        else:
            drug_loss   = 0
            
        if self.lambda_weight[2]:
            ddi_loss    = self.lambda_weight[2] * self.interact(pred_drug, real_diag, ddi_matrix)
        else:
            ddi_loss    = 0
            
        if self.lambda_weight[3]:
            pl_loss     = self.lambda_weight[3] * torch.mean(torch.abs(torch.sum(pred_drug > .5, dim = 1) - torch.sum(real_drug, dim = 1)).to(device))
        else:
            pl_loss     = 0
            
        if self.lambda_weight[4]:
            set_loss    = self.lambda_weight[4] * self.earth_movers_distance_loss((pred_drug > .5).type(torch.float), real_drug).to(device)
        else:
            set_loss    = 0
        
        if self.lambda_weight[5] and not att_score is None:
            att_score   = torch.sum(torch.sum(att_score, dim = -1), dim = 1)
            att_loss    = self.lambda_weight[5] * torch.mean((1 - att_score.masked_fill(real_drug == 0, 0) + att_score.masked_fill(real_drug == 1, 0)).view(-1))
        else:
            att_loss    = 0
            
        if self.lambda_weight[6]:
            len_loss    = self.lambda_weight[6] * torch.pow(torch.mean(torch.tensor([torch.sum(pred_drug[i]>.5) - len(real_drug[i]) for i in range(len(pred_drug))], dtype=torch.float)), 2)
        else:
            len_loss    = 0
            
#         print("cycle_loss:", cycle_loss)
#         print("drug_loss:", drug_loss)
#         print("ddi_loss:", ddi_loss)
#         print("pl_loss:", pl_loss)
#         print("set_loss:", set_loss)
#         print("len_loss:", len_loss)
        
        return cycle_loss + drug_loss + ddi_loss + pl_loss + set_loss + att_loss + len_loss


def padding_data(lses, max_length, fillin = -1, mode = 1):
    if mode:
        return list(lses) + [fillin] * (max_length - len(lses) - 1)
    return [list(ls) + [fillin] * (max_length - len(ls) - 1) for ls in lses]


class MyDataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''
        This function will return the index-selected content
        '''
        return self.df.loc[idx, "SUBJECT_ID"], \
            self.df.loc[idx, "HADM_ID"], \
            torch.tensor(padding_data(self.df.loc[idx, "ICD9_CODE"], syms_max + 2, fillin = SYM_NUM)), \
            torch.tensor(padding_data(self.df.loc[idx, "NDC"], med_max + 2, fillin = DRUG_NUM))
            
def get_recall(y_preds, y_trues):
    recall = []
    for y_pred, y_true in zip(y_preds, y_trues):
        intersection = len(set(y_pred).intersection(y_true))
        if len(y_true) != 0:
            recall.append(intersection / len(y_true))
        else:
            recall.append(0)
    return np.mean(recall)
    
def get_precision(y_preds, y_trues):
    precision = []
    cnt       = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        intersection = len(set(y_pred).intersection(y_true))
        if len(y_pred) != 0:
            precision.append(intersection / len(y_pred))
        else:
            precision.append(0)
        if precision[-1] < .5:
            bad_idx.append(cnt)
        cnt += 1
    return np.mean(precision)
    
def get_F1_score(y_preds, y_trues):
    f1_score = []
    for y_pred, y_true in zip(y_preds, y_trues):
        intersection = len(set(y_pred).intersection(y_true))
        if len(y_pred) != 0:
            precision = intersection / len(y_pred)
        else:
            precision = 0
            
        if len(y_true) != 0:
            recall = intersection / len(y_true)
        else:
            recall = 0
            
        if recall + precision == 0:
            f1_score.append(0)
        else:
            f1_score.append((2 * precision * recall) / (recall + precision))
    return np.mean(f1_score)


def onehot2ids(x):
    res = []
    for t in x:
        res.append((np.where(np.array(t) > 0.5)[0]).tolist())
    return res

def cu_sum(number):
    return number * (1+number)/2

def ddi_rate_calculation(ddi_table, d_list):
    count = 0
    pairs = cu_sum(len(d_list)-1)
    
    if pairs == 0:
        return 0
    
    for i1 in range(len(d_list)-1, -1, -1):
        if i1 == 0:
            break
        else:
            for i2 in range(i1-1, -1, -1):
                if ddi_table[i1][i2] == 1:
                    count += 1
    return count/pairs

def jaccard_score_manual(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1).union(list2))
    jaccard_score = intersection / union
    return jaccard_score