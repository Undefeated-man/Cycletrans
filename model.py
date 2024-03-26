import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill

from utils import get_param, OneHotEncoder, Embedder
from tqdm import tqdm, trange
from torch.cuda import amp
from collections import OrderedDict
from transformers import get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

# initialize
USE_CUDA = torch.cuda.is_available()
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


class Attention(nn.Module):
    def __init__(self, 
                 num_head = 8, 
                 d_model  = 64,
                 dropout  = .2):
        super(Attention, self).__init__()
        self.num_head    = num_head
        self.d_model     = d_model
        self.w_k         = get_param((num_head, d_model, d_model))
        self.b_k         = get_param((num_head, 1, d_model))
        self.w_q         = get_param((num_head, d_model, d_model))
        self.b_q         = get_param((num_head, 1, d_model))
        self.w_v         = get_param((num_head, d_model, d_model))
        self.b_v         = get_param((num_head, 1, d_model))
        self.w_o         = get_param((num_head * d_model, d_model))
        self.b_o         = get_param((1, d_model))
        self.dropout     = nn.Dropout(dropout)
        self.softmax     = nn.Softmax(dim = -1)
    
#     def translate(self, x, reverse = False):
#         if reverse:
#             # reverse back to the original shape
# #             x = x.reshape(-1, self.num_head, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)
#             # [bs, seq_len, dim, num_head]
#             return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
#         else:
#             return x.reshape(x.shape[0], x.shape[1], self.num_head, -1).permute(0, 2, 1, 3)
        
    def translate(self, x, reverse = False):
        if reverse:
            # reverse back to the original shape
#             x = x.reshape(-1, self.num_head, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)
            # [bs, num_head, seq_len, dim]
            x = x.permute(0, 2, 1, 3)    
            return x.reshape(x.shape[0], x.shape[1], -1)
        else:
            # [bs, num_head, seq_len, dim]
            return torch.stack([x.clone() for _ in range(self.num_head)]).permute(1, 0, 2, 3)
#             return x.reshape(x.shape[0], x.shape[1], self.num_head, -1).permute(0, 2, 1, 3)
    
    def cross_att(self, q, k, mask = None, seg_ids = None):
        if seg_ids is None:
            seg_ids = torch.tensor([1] * q.shape[1], device = q.device)
        else:
            seg_ids = seg_ids.to(q.device)
        q = torch.matmul(self.translate(q), self.w_q) + self.b_q
        v = torch.matmul(self.translate(k), self.w_v) + self.b_v 
        k = torch.matmul(self.translate(k), self.w_k) + self.b_k 
        att_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_model) 
        if not mask is None: 
            # att_score = torch.where(mask.unsqueeze(1).type(torch.bool), att_score, torch.tensor(-torch.inf)) 
            att_score = att_score.masked_fill(mask.unsqueeze(1) == 0, -1e9) 
        att_score = self.dropout(self.softmax(att_score)) 
        out = torch.matmul(att_score, v) 
        out = torch.matmul(self.translate(out, True), self.w_o) + self.b_o 
        return out.squeeze(-1), att_score
    
    def forward(self, x, mask = None):
        q         = torch.matmul(self.translate(x), self.w_q) + self.b_q
        k         = torch.matmul(self.translate(x), self.w_k) + self.b_k
        v         = torch.matmul(self.translate(x), self.w_v) + self.b_v
        att_score = torch.matmul(q, k.transpose(-1, -2)) /  np.sqrt(self.d_model)
        
        if not mask is None:
#             att_score = torch.where(mask.unsqueeze(1).type(torch.bool), att_score, torch.tensor(-torch.inf))
            att_score = att_score.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        # att_score: [bs, num_head, seq_len, seq_len]
        att_score = self.dropout(self.softmax(att_score))
        out       = torch.matmul(att_score, v)
        out = torch.matmul(self.translate(out, True), self.w_o) + self.b_o 
        return out.squeeze(-1)


class EncoderBlock(nn.Module):
    """ Encoder Block in Pre-LN Transformer block"""
    def __init__(self, 
                 num_head = 8, 
                 max_len  = 64, 
                 d_model  = 64,
                 dropout  = .2,
                 position_encode = False):
        super(EncoderBlock, self).__init__()
        if position_encode:
            self.embedder    = Embedder(max_len, d_model)
        else:
            self.embedder    = None
        self.multi_att   = Attention(num_head, d_model, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(OrderedDict([
            ("linear layer1",   nn.Linear(d_model, 2 * d_model)),
            ("activation func", nn.ReLU()),
            ("linear layer2",   nn.Linear(2 * d_model, d_model))
        ]))
        
    def forward(self, x):
        # do a pre-normalization
        x, mask, seg_ids  = x[0], x[1], x[2]
        if not self.embedder is None:
            x    = self.embedder(x)
        if seg_ids is None:
            seg_ids = torch.ones((x.shape[1], x.shape[2]), device = x.device)
        x        = self.norm1(x + seg_ids)
        out1     = self.multi_att(x, mask)
        out2     = self.norm2(out1 + x)
        out3     = self.feedforward(out2) + out2
        return out3, mask, seg_ids


hist_max_len = 176
HIDDEN_DIM = None

def pad(embd, max_len, mask = False):
    if mask:
        mask = [torch.cat([torch.ones((embd[i].shape[0]), 1), \
                torch.zeros((max_len - embd[i].shape[0]), 1)]) for i in range(len(embd))]
        return [torch.cat([embd[i], torch.zeros((max_len - embd[i].shape[0]), HIDDEN_DIM)]) for i in range(len(embd))], mask
    return [torch.cat([embd[i], torch.zeros((max_len - embd[i].shape[0]), HIDDEN_DIM)]) for i in range(len(embd))]

def sig2embd(sig, weight, max_len = None, mask = None):
    """
        The function to convert the sigmoid encoding to the original encoding.
    """
    res = []
    idx = (sig > .5).cpu()
    if max_len is None:
        for i in range(len(sig)):
            res.append(torch.mean(weight[idx[i]], dim = 0))
        return torch.stack(res, dim = 0)
    else:
        # provide the embedding without reducing the dimension
        for i in range(len(sig)):
            res.append(weight[idx[i]])
        if mask:
            tmp = pad(res, max_len, mask)
            return torch.stack(tmp[0], dim = 0), tmp[1]
        else:
            return torch.stack(pad(res, max_len), dim = 0)


class CycleSafeDrugTransformer(nn.Module):
    """
        CycleSafeDrugTransformer is a model which considers all the information of the patient, including the symptoms, all the hitorical symptoms, and suggests the drugs.

        Args:
            - syms_embd: A path of a .pt file which saved the initial syms_embeddings (e.g., "./syms_embd.pt")
            - num_drugs: the number of the drugs
            - hidden_dim: the hidden dimension of the model
            - n_blocks: the number of the transformer blocks
            - aggrr_type: the type of the aggregation function, which can be "sum", "mean", "max"

        Returns:
            - The result of model computation.
    """
    def __init__(self, syms_embd = None, sub2diag_dict = None, ddi_mat = None,
                num_drugs = DRUG_NUM, num_sym = SYM_NUM, d_model = 256,
                hidden_dim = 256, n_blocks = 3, dropout = .1, aggr_type = "sum", cpu = False):
        super(CycleSafeDrugTransformer, self).__init__()

        if cpu:
            self.device    = torch.device("cpu")
        else:
            self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 
        self.num_drug      = num_drugs
        self.sub2diag      = sub2diag_dict
        self.syms_embd     = nn.Embedding(num_sym + 1, d_model)
        if syms_embd:
            tmp            = torch.load(syms_embd) 
            self.syms_embd.weight.data.copy_(tmp)
        self.drug_embd     = nn.Embedding(num_drugs + 1, d_model)
#         self.ln1           = nn.Linear(d_model, hidden_dim, bias = True)
        self.outln         = nn.Linear(d_model, 1, bias = True)
#         self.transln       = nn.Linear(SYM_NUM, num_drugs, bias = True)
    
        if not ddi_mat is None:
            self.ddi_att   = Attention(d_model = d_model, num_head = 1, dropout = dropout).to(self.device)
            self.ddi_ln    = nn.Linear(d_model, 1, bias = True)
            self.ddi_mask  = []
            self.ddi_mat   = torch.from_numpy(ddi_mat).float().to(self.device)
            self.ddi_mat.requires_grad = False
        else:
            self.ddi_mat   = None
            
        self.out_att       = Attention(d_model = d_model, num_head = 2, dropout = 0).to(self.device)    
        self.relu          = nn.ReLU()
        self.softmax       = nn.Softmax(dim = 1)
        self.sigmoid       = nn.Sigmoid()
        self.model         = nn.Sequential()
        self.aggr_type     = aggr_type
        self.hist_att      = Attention(d_model = d_model, dropout = dropout).to(self.device)
        self.sym_att       = Attention(d_model = d_model, dropout = dropout).to(self.device)
        self.layernorm     = nn.LayerNorm(d_model)
        self.sym_encoder  = OneHotEncoder(num_sym)

#         self.model.add_module(f"Transformer Block_{0}", EncoderBlock(max_len = DRUG_NUM, d_model = d_model, dropout = dropout, position_encode = False))
        for i in range(n_blocks):
            self.model.add_module(f"Transformer Block_{i}", EncoderBlock(max_len = DRUG_NUM, d_model = d_model, dropout = dropout))

    def pad(self, x, max_len):
        mask = torch.ones((max_len, 1))
        mask[x.shape[0]:] = 0
        return torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])]), mask

    def forward(self, syms_ids, sub_ids, hadm_ids):
        sym_embds      = []
        hist_diag_embd = []
        hist_mask      = []
        sym_mask       = []

        for i in range(len(syms_ids)):
            t_syms_id = syms_ids[i]
            t_syms_id = t_syms_id[:t_syms_id[-1]]
            try:
                t_s, t_ms = self.pad(self.syms_embd(t_syms_id), SYM_NUM)
            except:
                print(t_syms_id)
                raise AttributeError
            sym_embds.append(t_s)
            sym_mask.append(t_ms)
            t_h, t_mh = self.pad(self.syms_embd(
                            torch.tensor(list(self.sub2diag[int(sub_ids[i])][int(hadm_ids[i])]))
                        ), hist_max_len)
            hist_diag_embd.append(t_h)
            hist_mask.append(t_mh)
        
        sym_embds      = torch.stack(sym_embds, dim = 0).to(self.device)
        sym_mask       = torch.stack(sym_mask, dim = 0).to(self.device)
        hist_diag_embd = torch.stack(hist_diag_embd, dim = 0).to(self.device)
        hist_mask      = torch.stack(hist_mask, dim = 0).to(self.device)
        hist_mask      = torch.matmul(sym_mask, hist_mask.transpose(-1, -2))
        sym_mask       = torch.matmul(sym_mask, sym_mask.transpose(-1, -2))
#         tmp_embd       = torch.sum(sym_embds, dim = 1)   # [bs, hidden_dim]
        tmp_embd       = self.sym_att(sym_embds, sym_mask)   # [bs, num_sym, hidden_dim]
        tmp_embd, _    = self.hist_att.cross_att(tmp_embd, hist_diag_embd, hist_mask)
        seg_ids        = torch.ones((tmp_embd.shape[1], tmp_embd.shape[2]), device = self.device)
        input_         = (tmp_embd, sym_mask, seg_ids)
        
        out            = self.model(input_)[0]
        
        if not self.ddi_mat is None:
            out2, att  = self.ddi_att.cross_att(
                                out,
                                torch.matmul(
                                    self.drug_embd.weight.data.to(self.device).transpose(-1, -2)[:, :-1].repeat([len(out), 1, 1]), 
                                    self.ddi_mat
                                ).transpose(-1, -2)
                           )
            out2      += out
            out        = torch.matmul(
                            out,
                            self.drug_embd.weight.data.to(self.device).repeat([len(out), 1, 1])[:, :-1].transpose(-1, -2)
                                     )
        else:
#             out, att   = self.out_att.cross_att(
#                                 out,
#                                 self.drug_embd.weight.data.to(self.device)[:-1, :].repeat([len(out), 1, 1])
#                            )
#             out       += out
            out        = torch.matmul(
                            out,
                            self.drug_embd.weight.data.to(self.device).repeat([len(out), 1, 1])[:, :-1].transpose(-1, -2)
                                     )

        
#             tmp           = self.sigmoid(out)
#             out_, self.ddi_mask = sig2embd(tmp, self.drug_embd.weight.data.cpu(), 131, True)
#             out_          = out_.to(self.device)
#             out_          = torch.matmul(out_.transpose(-1, -2), self.ddi_mat).transpose(-1, -2)
#             self.ddi_mask = torch.stack(self.ddi_mask, dim = 0)
#             self.ddi_mask = (self.ddi_mask @ self.ddi_mask.transpose(-1, -2)).to(self.device)
#             out_          = self.ddi_ln(self.ddi_att(out_, self.ddi_mask)).squeeze(-1)
#             out           = out_ + out
        att            = out
        out            = torch.mean(out, dim = 1).squeeze(1)
#         out            = self.outln(out.transpose(-1, -2)).squeeze(-1)
        
        
#         real_diag      = self.diag_encoder(diag_ids.cpu()).to(self.device)
#         real_diag      = real_diag.view(real_diag.shape[0], -1, real_diag.shape[-1]).repeat((1, 131, 1))
#         out  = torch.matmul(out, self.diag_embd.weight.data.to(self.device).repeat([len(out), 1, 1]).transpose(-1, -2))
#         out  = self.softmax(out.masked_fill(real_diag == 0, -1e9))
#         out            = torch.matmul(out, out_).view(-1, self.num_drug)
        return out, att  # self.sigmoid(out), att
    
    
class Drug2Diagnosis(nn.Module):
    """
        Using the drug information to predict the symptoms. Using the cycleGAN's idea to train the model.
        These 2 models are shared the embeddings.
    """
    def __init__(self, sgdt_model = None, num_sym = SYM_NUM, hidden_dim = 256, 
                share = True, n_blocks = 3, dropout = .1, aggr_type = "sum", cpu = False):
        super(Drug2Diagnosis, self).__init__()

#         assert aggr_type in ["sum", "mean", "max"]
        if cpu:
            self.device    = torch.device("cpu")
        else:
            self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if share:
            self.drug_embd = sgdt_model.drug_embd
            self.syms_embd = sgdt_model.syms_embd
        else:
            self.drug_embd = sgdt_model.drug_embd.copy()
            self.syms_embd = sgdt_model.sym_embd.copy()
        self.model     = nn.Sequential()
#         self.ln1       = nn.Linear(hidden_dim, hidden_dim)
#         self.ln2       = nn.Linear(hidden_dim, 1)
#         self.transln   = nn.Linear(HIDDEN_DIM, hidden_dim, bias = True)
        self.relu      = nn.ReLU()
        self.softmax   = nn.Softmax(dim = 1)
        self.sigmoid   = nn.Sigmoid()
        self.aggr_type = aggr_type
        self.sym_att   = Attention(d_model = HIDDEN_DIM, dropout = dropout).to(self.device)

        for i in range(n_blocks):
            self.model.add_module(f"Transformer Block_{i}", EncoderBlock(max_len = self.drug_embd.weight.shape[0] + 1, 
                                                                             d_model = HIDDEN_DIM, dropout = dropout))

    def pad(self, x, max_len, mask = False):
        if mask:
            mask = torch.ones((max_len, 1))
            mask[x.shape[0]:] = 0
            return torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])]), mask
        else:
            return torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])])

    def forward(self, drug_embd, drug_mask, sym_ids):
        syms_embd      = []
        syms_mask      = []
        # drug_embd      = []

        for i in range(len(sym_ids)):
            t_syms_id  = sym_ids[i]
            t_syms_id  = t_syms_id[:t_syms_id[-1]]
            t_s, t_ms  = self.pad(self.syms_embd(t_syms_id), SYM_NUM, True)
            syms_embd.append(t_s)
            syms_mask.append(t_ms)
            # t_drug_id = drug_ids[i]
            # t_drug_id = t_drug_id[:t_drug_id[-1]]
            # drug_embd.append(self.pad(self.drug_embd(t_drug_id), 58))
        
        syms_embd      = torch.stack(syms_embd, dim = 0).to(self.device)
        syms_mask      = torch.stack(syms_mask, dim = 0).to(self.device)
        syms_mask      = torch.matmul(syms_mask, syms_mask.transpose(-1, -2))
        # drug_embd      = torch.stack(drug_embd, dim = 1).to(self.device)
        drug_embd      = drug_embd.to(self.device)
        drug_mask      = drug_mask.to(self.device)
#         tmp_embd       = torch.sum(self.sym_att(syms_embd, syms_mask), dim = 1)   # [bs, hidden_dim]
# #         tmp_embd       = torch.mean(syms_embd, dim = 1)   # [bs, hidden_dim]

#         # Do the aggregation
#         drug_emb_shape = drug_embd.shape
#         if self.aggr_type == "sum":
# #             tmp_embd   = torch.sum(torch.cat([tmp_embd.reshape(drug_emb_shape[0], -1, drug_emb_shape[-1]), drug_embd], dim = 1), dim = 1)
#             tmp_embd   = torch.cat([tmp_embd.reshape(drug_emb_shape[0], -1, drug_emb_shape[-1]), drug_embd], dim = 1)
#         elif self.aggr_type == "mean":
#             tmp_embd   = torch.mean(torch.cat([tmp_embd.reshape(drug_emb_shape[0], -1, drug_emb_shape[-1]), drug_embd], dim = 1), dim = 1)
#         elif self.aggr_type == "max":
#             tmp_embd   = torch.max(torch.cat([tmp_embd.reshape(drug_emb_shape[0], -1, drug_emb_shape[-1]), drug_embd], dim = 1), dim = 1)
        out            = self.model((drug_embd, drug_mask, None))
#         out            = self.model((tmp_embd.reshape(tmp_embd.shape[0], -1, tmp_embd.shape[-1]), drug_mask, None))
        out, att       = self.sym_att.cross_att(out[0], syms_embd.repeat(1, 1, 1))
#         print(out.shape)
#         out            = torch.sum(self.transln(out),  dim = 1)

        out            = torch.matmul(
                            out,
                            self.syms_embd.weight.data.to(self.device).repeat([len(out), 1, 1])[:, :-1].transpose(-1, -2)
                                     )
#         out            = torch.matmul(out.unsqueeze(1).transpose(-1, -2), att)
    
#         out            = self.relu(self.ln2(out.transpose(-1, -2)))
#         print(out.shape)
        out            = torch.mean(out, dim = 1).squeeze(1)
#         out            = self.sigmoid(torch.mean(self.ln2(out.transpose(-1, -2)).squeeze(-1), dim = 1))
        return out