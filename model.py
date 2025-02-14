import torch
from torch import nn
import os
from torch_geometric.nn import global_add_pool,global_mean_pool,SAGPooling,global_max_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
import torch.nn.functional as F
import dgl.function as fn
import numpy as np

from model_graph import StructureEncoder


class SequenceEncoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.smi_embed = nn.Embedding(65, self.hidden_dim, padding_idx=0)
        self.smi_attention = nn.MultiheadAttention(self.hidden_dim, 4)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim, out_channels=40, kernel_size=4),
            # nn.ReLU(),
            nn.Conv1d(in_channels=40, out_channels=40 * 2, kernel_size=6),
            # nn.ReLU(),
            nn.Conv1d(in_channels=40 * 2, out_channels=40 * 4, kernel_size=8),
            # nn.ReLU(),
        )
        # self.Drug_CNNs = nn.Sequential(
        #     nn.Conv1d(in_channels=hidden_dim, out_channels=40, kernel_size=4, groups=hidden_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=40, out_channels=40 * 2, kernel_size=6, groups=40),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=40 * 2, out_channels=40 * 4, kernel_size=8, groups=40 * 2),
        #     nn.ReLU(),
        # )
        self.Drug_max_pool = nn.MaxPool1d(85)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc = nn.Linear(160, hidden_dim)

    def PositionalEncoding(self, num_hiddens, max_len, embedding):
        P = torch.zeros((max_len, num_hiddens)).to('cuda:0')
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(max_len, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        P[:, 0::2] = torch.sin(X.to('cuda:0'))
        P[:, 1::2] = torch.cos(X.to('cuda:0'))
        embedding = embedding + P[:embedding.shape[0], :]
        return embedding
    def forward(self, smi):
        smi_emb = self.smi_embed(smi)  # [512,100,64]
        # print(smi_emb.shape)
        smi_emb = self.PositionalEncoding(self.hidden_dim, 100, smi_emb)
        # smi_emb, atte = self.smi_attention(smi_emb, smi_emb, smi_emb)
        # print(smi_emb.shape)
        smi_emb = smi_emb.permute(0, 2, 1)  # [512,64,100]
        smi_emb = self.Drug_CNNs(smi_emb)  # [512,160,85]
        smi_emb = self.Drug_max_pool(smi_emb).squeeze(2)
        file_path = 's.npy'
        si = smi_emb.cpu()
        # print(sy.shape)
        # 检查文件是否存在，存在则加载并追加，不存在则新建
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            updated_data = np.vstack([existing_data, si])
        else:
            updated_data = si

        # 保存到文件
        np.save(file_path, updated_data)
        smi_emb = self.dropout(smi_emb)
        smi_emb = self.leaky_relu(self.fc(smi_emb))
        # file_path = 's.npy'
        # si = smi_emb.cpu()
        # # print(sy.shape)
        # # 检查文件是否存在，存在则加载并追加，不存在则新建
        # if os.path.exists(file_path):
        #     existing_data = np.load(file_path)
        #     updated_data = np.vstack([existing_data, si])
        # else:
        #     updated_data = si
        #
        # # 保存到文件
        # np.save(file_path, updated_data)
        return smi_emb


class FingerEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(FingerEncoder, self).__init__()
        self.num_features = 1024
        self.hidden_dim_lstm = 256*2
        self.fc1 = nn.Linear(self.num_features, self.hidden_dim_lstm)
        self.dropout = nn.Dropout(0.1)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim_lstm, hidden_dim)

    def forward(self, ecfp):

        fpn_out = self.fc1(ecfp)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        file_path = 'fp.npy'
        fi = fpn_out.cpu()
        # print(sy.shape)
        # 检查文件是否存在，存在则加载并追加，不存在则新建
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            updated_data = np.vstack([existing_data, fi])
        else:
            updated_data = fi

        # 保存到文件
        np.save(file_path, updated_data)
        fpn_out = self.fc2(fpn_out)
        # file_path = 'fp.npy'
        # fi = fpn_out.cpu()
        # # print(sy.shape)
        # # 检查文件是否存在，存在则加载并追加，不存在则新建
        # if os.path.exists(file_path):
        #     existing_data = np.load(file_path)
        #     updated_data = np.vstack([existing_data, fi])
        # else:
        #     updated_data = fi
        #
        # # 保存到文件
        # np.save(file_path, updated_data)
        return fpn_out


class MFDL_DDI(nn.Module):
    def __init__(self, in_dim, edge_in_dim, n_iter, hidden_dim=128):
        super().__init__()
        self.sequ_encoding = SequenceEncoder(hidden_dim*2)
        self.finger_encoding = FingerEncoder(hidden_dim*2)
        self.graph_encoding = StructureEncoder(hidden_dim, in_dim, edge_in_dim, n_iter)

        self.self_attn1 = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=2, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim*2)

        # self.weight_g = nn.Parameter(torch.randn(1, hidden_dim*2))
        # self.weight_s = nn.Parameter(torch.randn(1, hidden_dim*2))
        # self.weight_f = nn.Parameter(torch.randn(1, hidden_dim*2))
        self.weight_g = nn.Parameter(torch.randn(1))
        self.weight_s = nn.Parameter(torch.randn(1))
        self.weight_f = nn.Parameter(torch.randn(1))

        self.mix_layer = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=2, dropout=0.1)

        self.rmodule = nn.Embedding(86, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, hidden_dim * 2),
            nn.PReLU(),
            # nn.Linear(hidden_dim * 2, hidden_dim * 2),
            # nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    def Fusion(self, smi_emb, fp_emb, graph_emb):
        # CF = torch.concat((smi_emb, fp_emb), -1)
        weights = torch.softmax(torch.cat([self.weight_g, self.weight_s, self.weight_f], dim=0), dim=0)
        w_g, w_s, w_f= weights[0], weights[1],  weights[2]

        # Calculate the combined feature H
        CF = w_g * graph_emb + w_s * smi_emb+w_f * fp_emb
        # CF = torch.concat((smi_emb, fp_emb, graph_emb), -1)
        # if sa_fe:
        #     features = CF
        #     features_cpu = features.cpu()
        #     labels = np.array(['SMILES'] * smi_emb.shape[0] + ['2D Graph'] * smi_emb.shape[0] + ['Fingerprint'] * smi_emb.shape[0])
        #     # 保存到文件
        #     np.save("features.npy", features_cpu.numpy())
        #     np.save("labels.npy", labels)
        # CF = CF.view(smi_emb.shape[0], 128, 3).permute(2, 0, 1)
        # MF = torch.mean(CF, dim=0, keepdim=True)
        # da, attn_weights = self.self_attn1(MF, CF, CF)
        # da = self.norm1(da)
        # outputs = da.squeeze(0)
        return CF
    def forward(self, head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_smi, tail_smi, head_fp, tail_fp, sa_fe):
        # sequence encoder
        h_smi_emb = self.sequ_encoding(head_smi)
        t_smi_emb = self.sequ_encoding(tail_smi)
        h_fp_emb = self.finger_encoding(head_fp)
        t_fp_emb = self.finger_encoding(tail_fp)
        h_graph_emb = self.graph_encoding(head_pairs, head_pairs_dgl, batch_h_e)
        t_graph_emb = self.graph_encoding(tail_pairs, tail_pairs_dgl, batch_t_e)


        h1 = self.Fusion(h_smi_emb, h_fp_emb, h_graph_emb)
        t1 = self.Fusion(t_smi_emb, t_fp_emb, t_graph_emb)

        #cross_attention
        h_att = self.mix_layer(t1, h1, h1)[0]
        t_att = self.mix_layer(h1, t1, t1)[0]
        h = h1 * 0.5 + h_att * 0.5
        t = t1 * 0.5 + t_att * 0.5

        pair = torch.cat([h, t], dim=-1)
        pair = self.lin(pair)
        rfeat = self.rmodule(rel)

        score = (pair * rfeat).sum(-1)

        # if sa_fe:
        #     features = pair
        #     features_cpu = features.cpu()
        #     labels = rfeat
        #     # 保存到文件
        #     np.save("pairs.npy", features_cpu.numpy())
        #     np.save("labels.npy", labels)
        if sa_fe:
            return score, pair, h1, t1
        else:
            return score