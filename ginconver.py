
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from torch_geometric.nn import GATConv, GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn import LSTM, Linear, Dropout, Softmax


N_HEADS  = 4
LSTM_OUT = 256

class FCNet(nn.Module):

    def __init__(self, dims, act='ReLU', dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            if i < len(dims) - 2:
                layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BANLayer(nn.Module):

    def __init__(self, v_dim, q_dim, h_dim, h_out=2, act='ReLU', dropout=0.2, k=3):
        super().__init__()
        self.k     = k
        self.h_dim = h_dim
        self.h_out = h_out
        self.c     = 32

        self.v_net = FCNet([v_dim, h_dim * k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * k], act=act, dropout=dropout)

        if k > 1:
            self.p_net = nn.AvgPool1d(k, stride=k)

        if h_out <= self.c:
            self.h_mat  = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * k).normal_() * 0.01)
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).zero_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * k, h_out), dim=None)

        self.bn      = nn.BatchNorm1d(h_dim)
        self.dropout = nn.Dropout(dropout)

    def attention_pooling(self, v, q, att_map):
        fusion = torch.einsum('bvk,bvq,bqk->bk', v, att_map, q)
        if self.k > 1:
            fusion = self.p_net(fusion.unsqueeze(1)).squeeze(1) * self.k
        return fusion

    def forward(self, v, q, softmax=True):
        v_num, q_num = v.size(1), q.size(1)

        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq',
                                    self.h_mat, v_, q_) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)
            v_ = self.v_net(v)
            q_ = self.q_net(q)

        if softmax:
            p = F.softmax(att_maps.view(-1, self.h_out, v_num * q_num), dim=2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(v_, q_, att_maps[:, i, :, :])

        logits = self.bn(logits)
        return logits, att_maps


# ─────────────────────────────────────────────────────────────────────────────
# BiCrossAttention + CoAttentionLayer
# ─────────────────────────────────────────────────────────────────────────────
class BiCrossAttention(nn.Module):

    def __init__(self, dim: int = 128, out_dim: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim  n_heads 整除"
        self.n_heads = n_heads
        self.d_head  = out_dim // n_heads
        self.scale   = self.d_head ** -0.5

        self.a_proj = nn.Linear(dim, out_dim)
        self.b_proj = nn.Linear(dim, out_dim)

        self.a_to_b_q = nn.Linear(out_dim, out_dim)
        self.a_to_b_k = nn.Linear(out_dim, out_dim)
        self.a_to_b_v = nn.Linear(out_dim, out_dim)
        self.b_to_a_q = nn.Linear(out_dim, out_dim)
        self.b_to_a_k = nn.Linear(out_dim, out_dim)
        self.b_to_a_v = nn.Linear(out_dim, out_dim)

        self.a_out = nn.Linear(out_dim, out_dim)
        self.b_out = nn.Linear(out_dim, out_dim)
        self.a_norm = nn.LayerNorm(out_dim)
        self.b_norm = nn.LayerNorm(out_dim)
        self.a_ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.b_ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.a_norm2 = nn.LayerNorm(out_dim)
        self.b_norm2 = nn.LayerNorm(out_dim)

        self.a_score = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.Tanh(),
            nn.Linear(out_dim // 2, 1)
        )
        self.b_score = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.Tanh(),
            nn.Linear(out_dim // 2, 1)
        )
        self.a_pool_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.b_pool_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

    def _cross(self, query, key_value, q_proj, k_proj, v_proj, out_proj,
               query_mask=None, key_mask=None):
        B, N_q, _ = query.shape
        Q = self._split_heads(q_proj(query))
        K = self._split_heads(k_proj(key_value))
        V = self._split_heads(v_proj(key_value))
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            attn = attn.masked_fill(~key_mask[:, None, None, :], -9e15)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        out = out_proj(out)
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).float()
        return out, attn

    def _attentive_max_pool(self, x, score_layer, pool_proj, mask=None):
        logits = score_layer(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -9e15)
        alpha = F.softmax(logits, dim=-1)
        alpha = self.dropout(alpha)
        attn_pool = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)

        if mask is not None:
            masked_x = x.masked_fill(~mask.unsqueeze(-1), -9e15)
            max_pool = masked_x.max(dim=1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        else:
            max_pool = x.max(dim=1).values
        return pool_proj(torch.cat([attn_pool, max_pool], dim=-1)), alpha

    def forward(self, a, b, a_mask=None, b_mask=None, return_attn=False):
        a0 = self.a_proj(a)
        b0 = self.b_proj(b)

        a_ctx, attn_a_to_b = self._cross(
            a0, b0,
            self.a_to_b_q, self.a_to_b_k, self.a_to_b_v, self.a_out,
            query_mask=a_mask, key_mask=b_mask
        )
        b_ctx, attn_b_to_a = self._cross(
            b0, a0,
            self.b_to_a_q, self.b_to_a_k, self.b_to_a_v, self.b_out,
            query_mask=b_mask, key_mask=a_mask
        )

        a_ctx = self.a_norm(a0 + a_ctx)
        b_ctx = self.b_norm(b0 + b_ctx)
        a_ctx = self.a_norm2(a_ctx + self.a_ffn(a_ctx))
        b_ctx = self.b_norm2(b_ctx + self.b_ffn(b_ctx))

        a_vec, a_alpha = self._attentive_max_pool(a_ctx, self.a_score, self.a_pool_proj, a_mask)
        b_vec, b_alpha = self._attentive_max_pool(b_ctx, self.b_score, self.b_pool_proj, b_mask)

        if return_attn:
            return a_vec, b_vec, {
                "a_to_b": attn_a_to_b,
                "b_to_a": attn_b_to_a,
                "a_pool": a_alpha,
                "b_pool": b_alpha,
            }
        return a_vec, b_vec



class CoAttentionLayer(nn.Module):


    def __init__(self, dim=128, num_heads=4, dropout=0.1, k=196):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q 投影（从图通路特征生成查询）
        self.v0_q_proj = nn.Linear(dim, dim)
        self.q0_q_proj = nn.Linear(dim, dim)

        # K, V 投影（从序列特征生成键和值）
        self.v1_kv_proj = nn.Linear(dim, dim * 2)
        self.q1_kv_proj = nn.Linear(dim, dim * 2)

        # 输出投影
        self.out_v_proj = nn.Linear(dim, dim)
        self.out_q_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)



    def _split_heads(self, x):
        B = x.size(0)
        return x.view(B, self.num_heads, self.head_dim)

    def _attention(self, query, key, value):
        attn = (query @ key.transpose(-2, -1)) * self.scale  # [B, h, h]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ value  # [B, h, d]
        return out

    def forward(self, v0, q0, v1, q1):
        B = v0.size(0)

        Q_v = self._split_heads(self.v0_q_proj(v0))   # [B, h, d_h]
        K_v, V_v = self.v1_kv_proj(v1).chunk(2, dim=-1)
        K_v = self._split_heads(K_v)
        V_v = self._split_heads(V_v)

        Q_q = self._split_heads(self.q0_q_proj(q0))
        K_q, V_q = self.q1_kv_proj(q1).chunk(2, dim=-1)
        K_q = self._split_heads(K_q)
        V_q = self._split_heads(V_q)

        attn_v = self._attention(Q_v, K_v, V_v)
        attn_q = self._attention(Q_q, K_q, V_q)

        out_v = self.out_v_proj(attn_v.reshape(B, self.dim))
        out_q = self.out_q_proj(attn_q.reshape(B, self.dim))


        m0 = v0 * q0

        return torch.cat([out_v, out_q, m0], dim=1)


class FusionRefine(nn.Module):

    def __init__(self, in_dim, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# GatedFusionLayer
# ─────────────────────────────────────────────────────────────────────────────
class GatedFusionLayer(nn.Module):
    def __init__(self, v_dim, q_dim, output_dim=256, dropout_rate=0.2):
        super(GatedFusionLayer, self).__init__()
        self.v_transform = nn.Linear(v_dim, output_dim)
        self.q_transform = nn.Linear(q_dim, output_dim)
        self.gate_transform = nn.Linear(output_dim * 2, output_dim)
        self.activation = nn.Tanh()
        self.output_dim = output_dim

    def get_output_shape(self):
        return self.output_dim

    def forward(self, v, q):
        v_proj = self.activation(self.v_transform(v))
        q_proj = self.activation(self.q_transform(q))

        concat_proj = torch.cat([v_proj, q_proj], dim=1)
        gate = torch.sigmoid(self.gate_transform(concat_proj))

        gated_output = gate * v_proj + (1 - gate) * q_proj
        return gated_output



class LinkAttention(nn.Module):

    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.query   = Linear(input_dim, n_heads)
        self.n_heads = n_heads
        self.softmax = Softmax(dim=-1)

    def forward(self, x, masks):

        query = self.query(x).transpose(1, 2)             # [B, n_heads, L]
        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)
        a = self.softmax(e)                               # [B, n_heads, L]
        out = torch.matmul(a, x).sum(dim=1)              # [B, input_dim]
        return out, a


def generate_masks(adj, adj_sizes, n_heads):

    B, L = adj.shape[0], adj.shape[1]
    out  = torch.ones(B, L, device=adj.device)
    for e_id, length in enumerate(adj_sizes):
        out[e_id, int(length):] = 0
    return out.unsqueeze(1).expand(-1, n_heads, -1)      # [B, n_heads, L]


def generate_pooled_mask(lengths, pooled_len, device):

    lengths_t = torch.as_tensor(lengths, device=device, dtype=torch.float32)
    max_len = lengths_t.max().clamp(min=1.0)
    pooled_lengths = torch.ceil(lengths_t / max_len * pooled_len).long().clamp(min=1, max=pooled_len)
    pos = torch.arange(pooled_len, device=device).unsqueeze(0)
    return pos < pooled_lengths.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTMEncode
# ─────────────────────────────────────────────────────────────────────────────
class LSTMEncode(nn.Module):


    def __init__(self, out_seq_len, num_feat_in=128, num_feat_out=128,
                 rnn_layers=2, dr=0.2, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx

        self.dropout    = Dropout(dr)
        self.encode_rnn = LSTM(num_feat_in, num_feat_in, num_layers=rnn_layers,
                               batch_first=True, bidirectional=True, dropout=dr)
        self.fc_in      = Linear(num_feat_in, num_feat_out)
        self.fusion_t   = LinkAttention(LSTM_OUT, N_HEADS)

        self.adapt_pool = nn.AdaptiveAvgPool1d(out_seq_len)
        self.fc_seq     = Linear(LSTM_OUT, 128)    # [B, L_out, 256] → [B, L_out, 128]
        self.fc_global  = Linear(LSTM_OUT, 128)    # [B, 256] → [B, 128]

    def forward(self, x_embed, x_idx=None):

        if x_idx is not None:
            lengths = (x_idx != self.padding_idx).sum(dim=1).cpu().numpy().astype(int)
        else:
            lengths = (x_embed.abs().sum(dim=-1) > 1e-9).sum(dim=1).cpu().numpy().astype(int)

        xt        = self.fc_in(x_embed)                        # [B, L, 128]
        xt_out, _ = self.encode_rnn(xt)                        # [B, L, 256]

        masks       = generate_masks(xt_out, lengths, N_HEADS) # [B, n_heads, L]
        global_raw, attn = self.fusion_t(xt_out, masks)        # [B, 256]
        global_feat = self.fc_global(global_raw)               # [B, 128]

        # AdaptiveAvgPool → 固定长度
        seq_feat = self.adapt_pool(xt_out.transpose(1, 2)).transpose(1, 2)
        # [B, L_out, 256]
        seq_feat = self.fc_seq(seq_feat)                        # [B, L_out, 128]

        return lengths, seq_feat, masks, global_feat, attn



# ─────────────────────────────────────────────────────────────────────────────
# DrugGraphEncoder
# ─────────────────────────────────────────────────────────────────────────────
class DrugGraphEncoder(nn.Module):

    def __init__(self, in_dim=78, hidden_dim=128, out_dim=128,
                 max_nodes=64, dropout=0.2):
        super().__init__()
        self.max_nodes  = max_nodes
        self.conv1      = GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.conv2      = GCNConv(hidden_dim, hidden_dim)
        self.conv3      = GCNConv(hidden_dim, out_dim)
        self.pool       = SAGPooling(out_dim, ratio=0.8)
        self.norm1      = nn.BatchNorm1d(hidden_dim)
        self.norm2      = nn.BatchNorm1d(hidden_dim)
        self.norm3      = nn.BatchNorm1d(out_dim)
        self.dropout    = nn.Dropout(dropout)
        self.global_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch, return_node_index=False):

        x = self.dropout(F.elu(self.norm1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.norm2(self.conv2(x, edge_index))))
        x = F.relu(self.norm3(self.conv3(x, edge_index)))

        g_mean = global_mean_pool(x, batch)
        g_max  = global_max_pool(x, batch)
        global_feat = self.global_proj(torch.cat([g_mean, g_max], dim=1))

        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, batch=batch)
        node_feat, node_mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        node_feat = node_feat * node_mask.unsqueeze(-1).float()
        if return_node_index:
            node_index, _ = to_dense_batch(perm, batch, max_num_nodes=self.max_nodes, fill_value=-1)
            return node_feat, global_feat, node_mask, node_index
        return node_feat, global_feat, node_mask


# ─────────────────────────────────────────────────────────────────────────────
# ProteinGraphEncoder
# ─────────────────────────────────────────────────────────────────────────────
class ProteinGraphEncoder(nn.Module):

    def __init__(self, in_dim=54, hidden_dim=128, out_dim=128,
                 fixed_nodes=128, dropout=0.2):
        super().__init__()
        self.fixed_nodes = fixed_nodes
        self.conv1  = GCNConv(in_dim, hidden_dim)
        self.conv2  = GCNConv(hidden_dim, hidden_dim)
        self.conv3  = GCNConv(hidden_dim, out_dim)
        self.pool3  = SAGPooling(out_dim, ratio=0.8)
        self.norm1  = nn.BatchNorm1d(hidden_dim)
        self.norm2  = nn.BatchNorm1d(hidden_dim)
        self.norm3  = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.global_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch, return_node_index=False):

        x = self.dropout(F.relu(self.norm1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.norm2(self.conv2(x, edge_index))))
        x = F.relu(self.norm3(self.conv3(x, edge_index)))
        x, edge_index, _, batch, perm, _ = self.pool3(x, edge_index, batch=batch)

        g_mean = global_mean_pool(x, batch)
        g_max  = global_max_pool(x, batch)
        global_feat = self.global_proj(torch.cat([g_mean, g_max], dim=1))

        node_feat, node_mask = to_dense_batch(x, batch, max_num_nodes=self.fixed_nodes)
        node_feat = node_feat * node_mask.unsqueeze(-1).float()
        if return_node_index:
            node_index, _ = to_dense_batch(perm, batch, max_num_nodes=self.fixed_nodes, fill_value=-1)
            return node_feat, global_feat, node_mask, node_index
        return node_feat, global_feat, node_mask


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────
class GINConvNet(nn.Module):

    def __init__(self,
                 drug_node_dim=78,   drug_max_nodes=64,
                 pro_node_dim=54,    pro_fixed_nodes=256,
                 drug_vocab_size=63, drug_max_len=100,
                 drug_seq_len=50,
                 pro_vocab_size=26,  pro_max_len=850,
                 pro_seq_len=256,
                 d_model=128,
                 ban_h_dim=256, ban_h_out=2, ban_k=3,
                 cross_n_heads=4,
                 n_output=1, dropout=0.2):
        super().__init__()
        self.d_model = d_model

        tv_size = 26
        dv_size = 62 + 1

        # ── Embedding ────────────────────────────────────────────────────────
        self.drug_embedding = nn.Embedding(dv_size, 128, padding_idx=0)
        self.pro_embedding  = nn.Embedding(tv_size, 128, padding_idx=0)

        # ── 编码器 ────────────────────────────────────────────────────────────
        self.drug_graph_enc = DrugGraphEncoder(
            in_dim=drug_node_dim, hidden_dim=d_model, out_dim=d_model,
            max_nodes=drug_max_nodes, dropout=dropout
        )
        self.pro_graph_enc = ProteinGraphEncoder(
            in_dim=pro_node_dim, hidden_dim=d_model, out_dim=d_model,
            fixed_nodes=pro_fixed_nodes, dropout=dropout
        )
        self.drug_seq_enc = LSTMEncode(out_seq_len=drug_seq_len, padding_idx=0)
        self.pro_seq_enc  = LSTMEncode(out_seq_len=pro_seq_len,  padding_idx=0)
        self.drug_seq_enc1 = LSTMEncode(out_seq_len=drug_seq_len, padding_idx=0)
        self.pro_seq_enc1  = LSTMEncode(out_seq_len=pro_seq_len,  padding_idx=0)

        self.seq_ban = weight_norm(
            BANLayer(v_dim=d_model, q_dim=d_model, h_dim=ban_h_dim,
                     h_out=ban_h_out, k=ban_k, dropout=dropout),
            name='h_mat', dim=None
        )


        self.graph_bicross = BiCrossAttention(
            dim=d_model, out_dim=128, n_heads=cross_n_heads, dropout=dropout
        )
        self.seq_bicross = BiCrossAttention(
            dim=d_model, out_dim=128, n_heads=cross_n_heads, dropout=dropout
        )
        self.co_attention = CoAttentionLayer(dim=128, k=196)
        self.graph_inter_refine = FusionRefine(in_dim=256, out_dim=128, dropout=dropout)
        self.seq_inter_refine = FusionRefine(in_dim=256, out_dim=128, dropout=dropout)
        self.gd = FusionRefine(in_dim=128, out_dim=128, dropout=dropout)
        self.gp = FusionRefine(in_dim=128, out_dim=128, dropout=dropout)
        self.co_refine = FusionRefine(in_dim=384, out_dim=256, dropout=dropout)


        self.s_gate = GatedFusionLayer(v_dim=128, q_dim=128, output_dim=256)
        self.g_gate = GatedFusionLayer(v_dim=128, q_dim=128, output_dim=256)


        self.mlp = nn.Sequential(
            nn.Linear(1152, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_output)
        )

        self._init_weights()

    # ── 初始化 ────────────────────────────────────────────────────────────────
    def _init_weights(self):

        for name, m in self.named_modules():
            if hasattr(m, 'weight_g'):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 前向传播 ──────────────────────────────────────────────────────────────
    def forward(self, data_mol, d_frags, data_pro, return_attention=False):


        smi    = data_mol.drug_smiles    # [B, drug_max_len]
        target = data_pro.target         # [B, pro_max_len]

        # ── 1. 图编码 ─────────────────────────────────────────────────────────
        if return_attention:
            drug_node_feat, drug_graph_global, drug_node_mask, drug_node_index = self.drug_graph_enc(
                data_mol.x, data_mol.edge_index, data_mol.batch, return_node_index=True
            )
        else:
            drug_node_feat, drug_graph_global, drug_node_mask = self.drug_graph_enc(
                data_mol.x, data_mol.edge_index, data_mol.batch
            )


        if return_attention:
            pro_node_feat, pro_graph_global, pro_node_mask, pro_node_index = self.pro_graph_enc(
                data_pro.x, data_pro.edge_index, data_pro.batch, return_node_index=True
            )
        else:
            pro_node_feat, pro_graph_global, pro_node_mask = self.pro_graph_enc(
                data_pro.x, data_pro.edge_index, data_pro.batch
            )


        drug_s_encode = self.drug_embedding(smi)           # [B, L_d, 128]
        drug_lengths, drug_seq_feat, _, drug_s1, _ = self.drug_seq_enc(drug_s_encode, x_idx=smi)
        drug_lengths2, drug_seq_feat2, _, drug_s2, _ = self.drug_seq_enc1(drug_s_encode, x_idx=smi)


        pro_s_encode = self.pro_embedding(target)          # [B, L_p, 128]
        pro_lengths, pro_seq_feat, _, pro_s1, _ = self.pro_seq_enc(pro_s_encode, x_idx=target)
        pro_lengths2, pro_seq_feat2, _, pro_s2, _ = self.pro_seq_enc1(pro_s_encode, x_idx=target)

        drug_seq_mask = generate_pooled_mask(drug_lengths, drug_seq_feat.size(1), drug_seq_feat.device)
        pro_seq_mask = generate_pooled_mask(pro_lengths, pro_seq_feat.size(1), pro_seq_feat.device)
        drug_seq_cnn_feat = self.drug_seq_cnn(drug_s_encode, token_idx=smi, padding_idx=0)
        pro_seq_cnn_feat = self.pro_seq_cnn(pro_s_encode, token_idx=target, padding_idx=0)


        seq_ban_out, seq_ban_attn = self.seq_ban(drug_seq_feat, pro_seq_feat)

        drug_graph_global_token = drug_graph_global.unsqueeze(1)
        pro_graph_global_token = pro_graph_global.unsqueeze(1)
        drug_graph_global_mask = torch.ones(
            drug_graph_global_token.size(0),
            drug_graph_global_token.size(1),
            dtype=torch.bool,
            device=drug_graph_global_token.device
        )
        pro_graph_global_mask = torch.ones(
            pro_graph_global_token.size(0),
            pro_graph_global_token.size(1),
            dtype=torch.bool,
            device=pro_graph_global_token.device
        )
        if return_attention:
            drug_graph_inter, pro_graph_inter, graph_attn = self.graph_bicross(
                drug_graph_global_token,
                pro_graph_global_token,
                a_mask=drug_graph_global_mask,
                b_mask=pro_graph_global_mask,
                return_attn=True
            )
        else:
            drug_graph_inter, pro_graph_inter = self.graph_bicross(
                drug_graph_global_token,
                pro_graph_global_token,
                a_mask=drug_graph_global_mask,
                b_mask=pro_graph_global_mask
            )


        graph_inter_out = self.graph_inter_refine(torch.cat([drug_graph_inter, pro_graph_inter], dim=1))

        co_out = self.co_attention(
            pro_graph_inter, drug_graph_inter, pro_s2, drug_s2
        )
        co_out = self.co_refine(co_out)


        s_gate_out = self.s_gate(drug_s1, pro_s1)          # [B, 256]

        g_gate_out = self.g_gate(drug_graph_global, pro_graph_global)  # [B, 256]

        feat = torch.cat([
            seq_ban_out,
            graph_inter_out,

            co_out,
            s_gate_out,
            g_gate_out,
        ], dim=1)

        out = self.mlp(feat)   # [B, n_output]
        if return_attention:
            return out, {
                "seq_ban_attn": seq_ban_attn,             # [B, h_out, drug_seq_len, pro_seq_len]
                "drug_graph_global_to_pro_graph_global_attn": graph_attn["a_to_b"],  # [B, heads, 1, 1]
                "pro_graph_global_to_drug_graph_global_attn": graph_attn["b_to_a"],  # [B, heads, 1, 1]
                "drug_graph_global_pool_attn": graph_attn["a_pool"],                 # [B, 1]
                "pro_graph_global_pool_attn": graph_attn["b_pool"],                  # [B, 1]
                "drug_node_mask": drug_node_mask,         # [B, drug_nodes]
                "pro_node_mask": pro_node_mask,           # [B, pro_nodes]
                "drug_node_index": drug_node_index,       # [B, drug_nodes], pooled node -> original atom index
                "pro_node_index": pro_node_index,         # [B, pro_nodes], pooled node -> original residue index
                "drug_smiles_idx": smi,                   # [B, raw_drug_len]
                "protein_seq_idx": target,                # [B, raw_pro_len]
                "drug_seq_mask": drug_seq_mask,           # [B, drug_seq_len]
                "pro_seq_mask": pro_seq_mask,             # [B, pro_seq_len]
                "drug_seq_cnn_feat_len": drug_seq_cnn_feat.size(1),
                "protein_seq_cnn_feat_len": pro_seq_cnn_feat.size(1),
                "drug_seq_feat_len": drug_seq_feat.size(1),
                "protein_seq_feat_len": pro_seq_feat.size(1),
            }
        return out
