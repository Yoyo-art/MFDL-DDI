import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree, softmax as pyg_softmax

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import numpy as np



class LinearBlock(nn.Module):


    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats

        self.block = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
            # Layer 2 (Residual handled in forward)
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
            # Layer 3 (Residual handled in forward)
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
            # Layer 4 (Residual handled in forward)
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
            # Final
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        layers = list(self.block)

        out1 = layers[1](layers[0](x))  # lin1

        out2 = layers[4](layers[3](layers[2](out1)))  # lin2
        out2 = (out2 + out1) / 2  # Residual 1

        out3 = layers[7](layers[6](layers[5](out2)))  # lin3
        out3 = (out3 + out2) / 2  # Residual 2

        out4 = layers[10](layers[9](layers[8](out3)))  # lin4
        out4 = (out4 + out3) / 2  # Residual 3

        out5 = layers[13](layers[12](layers[11](out4)))  # lin5
        return out5


class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Linear(hidden_dim, 1)  # Simplified to Linear if strictly global attention

    def forward(self, x, edge_index, batch):
        scores = self.conv(x).squeeze(-1)
        scores = pyg_softmax(scores, batch)
        gx = global_add_pool(x * scores.unsqueeze(-1), batch)
        return gx


class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter
        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)

        glorot(self.a)
        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):
        edge_index = data.edge_index
        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)

        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr

        out_list = []
        gout_list = []

        # Line Graph Message Passing
        for n in range(self.n_iter):
            msg = scatter(out[data.line_graph_edge_index[0]], data.line_graph_edge_index[1],
                          dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + msg

            # Global Pooling over edges
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)

            out_list.append(out)
            gout_list.append(torch.tanh(self.lin_gout(gout)))

        # Aggregation
        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)

        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)

        deg = degree(data.edge_index_batch, dtype=torch.long)
        scores = scores.repeat_interleave(deg, dim=0)

        out = (out_all * scores).sum(-1)


        x = data.x + scatter(out, edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x


class DrugEncoder(nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim, n_iter):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)
        return x



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, g, h):
        # h: [N, d_in]

        # 1. Linear Projections
        Q_h = self.Q(h).view(-1, self.num_heads, self.out_dim)
        K_h = self.K(h).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(h).view(-1, self.num_heads, self.out_dim)

        g.ndata['Q'] = Q_h
        g.ndata['K'] = K_h
        g.ndata['V'] = V_h

        # 2. Calculate Edge Features & Attention Scores

        g.apply_edges(fn.u_mul_v('K', 'Q', 'score'))

        # Scaling
        g.edata['score'] = g.edata['score'] / np.sqrt(self.out_dim)

        # 3. Prepare e_out (Edge Output)

        # e_out = g.edata['score'].view(-1, self.out_dim * self.num_heads)


        # 4. Prepare Attention Weights (Scalar for Softmax)

        attn_score = g.edata['score'].sum(dim=-1, keepdim=True)


        attn_score = torch.clamp(attn_score, min=-5, max=5)

        # 5. Softmax (DGL Optimized)

        attn_weights = edge_softmax(g, attn_score)

        # 6. Aggregation
        # Message Passing: src_V * edge_attn -> sum -> dst_h
        g.edata['a'] = attn_weights
        g.update_all(fn.u_mul_e('V', 'a', 'm'), fn.sum('m', 'wV'))

        h_out = g.ndata['wV'].view(-1, self.out_dim * self.num_heads)

        return h_out


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias=False)

        self.O_h = nn.Linear(out_dim, out_dim)
        # self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_h = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
           

    def forward(self, g, h):
        h_in1= h

        # 1. Normalization (Pre-Norm style is often better, but keeping original Post/Pre logic)
        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)


        # 2. Attention
        h_attn= self.attention(g, h)

        h = self.O_h(h_attn)


        h = F.dropout(h, self.dropout, training=self.training)


        # 3. Residual
        if self.residual:
            h = h_in1 + h


        h_in2= h

        # 4. Normalization 2
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        # 5. FFN
        h = self.FFN_h(h)

        if self.residual:
            h = h_in2 + h

        return h


class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim, in_dim, edge_in_dim, n_iter, num_heads=2, n_layers=3, dropout=0.0):
        super().__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter)
        self.embedding_e = nn.Linear(edge_in_dim, hidden_dim)
        self.graph_pred_linear = nn.Identity()  # Placeholder if needed
        self.in_feat_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(n_layers)
        ])

    def fusion(self, sub, data):
        # Global Pooling
        # sub: [N, hidden_dim], data.batch: [N]
        Max = global_max_pool(sub, data.batch)
        Mean = global_mean_pool(sub, data.batch)
        d_g = torch.cat([Max, Mean], dim=-1)
        return d_g

    def forward(self, h_data, g, e):
        # 1. PyG MPNN Encoding
        s_h = self.drug_encoder(h_data)
        h = self.in_feat_dropout(s_h)

        # 2. Edge Embedding
        # e = self.embedding_e(e.float())

        # 3. DGL Transformer Layers
        for conv in self.layers:
            h= conv(g, h)

        # 4. Readout
        out = self.fusion(h, h_data)
        return out


