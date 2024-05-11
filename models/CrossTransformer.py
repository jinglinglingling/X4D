import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, q, k, v):
        attention_scores = torch.matmul(q, k.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights



class SnippetEmbedding(nn.Module):
    def __init__(self, n_head, d_model,k_dim, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.k_dim = k_dim

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False).cuda()
        self.w_qs.requires_grad = True
        self.w_ks = nn.Linear(k_dim, n_head * d_k, bias=False).cuda()
        self.w_vs = nn.Linear(k_dim, n_head * d_v, bias=False).cuda()
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = CrossAttention()
        self.layer_norm = nn.LayerNorm(d_model).cuda()
        self.fc = nn.Linear(n_head * d_v, d_model).cuda()
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn_weights = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output

if __name__ == '__main__':
    q = torch.randn(8, 150, 1024).cuda()
    k = torch.ones(8, 150, 512).cuda()
    v = torch.ones(8, 150, 512).cuda()
    model = SnippetEmbedding(1, 1024, 512, 512, 512)
    output = model(q,k,v)
    print(output)