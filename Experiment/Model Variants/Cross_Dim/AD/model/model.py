import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
import math
from math import sqrt
from typing import List


class ContinuousEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding

        # Create weight and bias as trainable parameters
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.d_embedding ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Input should be [B, {self.n_features}], but got {x.shape}")

        x = x.unsqueeze(-1)  
        out = x * self.weight + self.bias  
        return out


class CategoricalEmbeddings(nn.Module):
    def __init__(self, cardinalities: List[int], d_embedding: int, bias: bool = True):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(x, d_embedding) for x in cardinalities])
        self.bias = nn.Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x):
        x = torch.stack([self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))], dim=-2)
        if self.bias is not None:
            x = x + self.bias
        return x 


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, return_attention=False):
        super(FullAttention, self).__init__()
        self.scale      = scale
        self.dropout    = nn.Dropout(attention_dropout)
        self.return_attention =return_attention

    def forward(self, queries, keys, values, return_attention=False):   # 加入參數
        B, L, H, E  = queries.shape
        _, S, _, D  = values.shape
        scale       = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)         # [B, H, L, S]
        A = torch.softmax(scale * scores, dim=-1)                       # attention scores
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.return_attention:
            return V.contiguous(), A                                    # 返回注意力權重 A
        else:
            return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1, return_attention=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=attention_dropout, return_attention=return_attention)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.return_attention = return_attention

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H       = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        if self.return_attention:
            out, attention_scores = self.inner_attention(queries, keys, values)
            out = out.view(B, L, -1)
            return self.out_projection(out), attention_scores        
        else:
            out = self.inner_attention(queries, keys, values)
            out = out.view(B, L, -1)
            return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, attention_dropout=0.1, residual_dropout=0.1, ffn_dropout=0.1, return_attention=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.return_attention = return_attention
        self.d_model = d_model

        self.sta_attention = AttentionLayer(d_model, n_heads, attention_dropout=attention_dropout, return_attention=return_attention)
        self.dim_attention = AttentionLayer(d_model, n_heads, attention_dropout=attention_dropout, return_attention=return_attention)

        self.residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_dropout        = nn.Dropout(ffn_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        batch       = x.shape[0]
        dim_num     = x.shape[1]
        sta_num     = x.shape[2]
        d_model     = x.shape[3]
        
        # === Dimension-wise Attention FIRST ===
        dim_in = rearrange(x, 'b dim_num sta_num d_model -> (b sta_num) dim_num d_model')

        dim_in_norm = self.norm1(dim_in)

        dim_enc, dim_attn = self.dim_attention(dim_in_norm, dim_in_norm, dim_in_norm)
        dim_out = dim_in_norm + self.residual_dropout(dim_enc)
        dim_ffn = dim_out + self.ffn_dropout(self.MLP1(self.norm2(dim_out)))

        # # === Statistic-wise Attention SECOND ===
        # sta_in = rearrange(dim_ffn, '(b sta_num) dim_num d_model -> (b dim_num) sta_num d_model', b=batch)

        # sta_in_norm = self.norm3(sta_in)

        # sta_enc, sta_attn = self.sta_attention(sta_in_norm, sta_in_norm, sta_in_norm)
        # sta_out = sta_in_norm + self.residual_dropout(sta_enc)
        # sta_ffn = sta_out + self.ffn_dropout(self.MLP2(self.norm4(sta_out)))

        # final_out = rearrange(sta_ffn, '(b dim_num) sta_num d_model -> b dim_num sta_num d_model', b=batch)

        # if self.return_attention:
        #     return final_out, None, None
        # else:
        #     return final_out

        # === Newly Added for Ablation Study ===
        final_out = rearrange(dim_ffn, '(b sta_num) dim_num d_model -> b dim_num sta_num d_model', b=batch)

        if self.return_attention:
            return final_out, None, None
        else:
            return final_out
        

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.cls_proj   = nn.Linear(dim, dim)
        self.norm       = nn.LayerNorm(dim)

        self.attn               = AttentionLayer(dim, num_heads, attention_dropout=attention_dropout, return_attention=False)
        self.residual_dropout   = nn.Dropout(residual_dropout)

    def forward(self, cls_token, patch_tokens):
        
        cls_proj        = self.cls_proj(cls_token)
        cls_proj_norm   = self.norm(cls_proj)

        updated = self.attn(cls_proj_norm, patch_tokens, patch_tokens)
        output  = cls_proj + self.residual_dropout(updated)

        return output


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=1, attention_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.cls_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.blocks = nn.ModuleList([CrossAttentionBlock(dim=dim, num_heads=num_heads, attention_dropout=attention_dropout, residual_dropout=residual_dropout) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(dim)

    def forward(self, cls_token, patch_tokens):
        x = self.cls_proj(cls_token)

        for block in self.blocks:
            x = block(x, patch_tokens)

        x = self.norm(self.out_proj(x))
        return x

class StackedTwoStageAttention(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int = None,
                 attention_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 return_attention   = False):
        super().__init__()

        self.num_layers         = num_layers
        self.return_attention   = return_attention

        self.layers = nn.ModuleList([
            TwoStageAttentionLayer(
                d_model             = d_model,
                n_heads             = n_heads,
                d_ff                = d_ff,
                attention_dropout   = attention_dropout,
                residual_dropout    = residual_dropout,
                ffn_dropout         = ffn_dropout,
                return_attention    = return_attention
            ) for _ in range(num_layers)
        ])

        self.cross = CrossAttention(
            dim                 = d_model,
            num_heads           = n_heads,
            num_layers          = 1,
            attention_dropout   = attention_dropout,
            residual_dropout    = residual_dropout
        )

        self.sta_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dim_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        batch, dim_num, sta_num, d_model = x.shape
        sta_attentions, dim_attentions = [], []
        
        dim_token = self.dim_token.expand(batch, -1, x.shape[2], -1)
        x = torch.cat([dim_token, x], dim=1)

        # sta_token = self.sta_token.expand(batch, x.shape[1], -1, -1)
        # x = torch.cat([sta_token, x], dim=2)
                      
        for i, layer in enumerate(self.layers):
            x, s_attn, d_attn = layer(x)
            
            # dim_token = x[:, 0, :, :]
            # sta_token = x[:, :, 0, :]
            
            # updated_dim_token = self.cross(dim_token, sta_token)
            # x[:, 0, :, :] = updated_dim_token

        if self.return_attention:
            return x, sta_attentions, dim_attentions
        else:
            return x
    

class DualTransformer(nn.Module):
    def __init__(self, 
                 ori_cardinalities: List[int],
                 num_features: int,
                 cat_features: int,
                 statistic_features: int,
                 dim_model: int,
                 num_heads: int,
                 dim_ff: int,
                 num_layers_cross: int,
                 num_labels: int,
                 return_attention: bool,
                 att_dropout: float,
                 res_dropout: float,
                 ffn_dropout: float):
        super().__init__()

        # === Embedding ===
        self.has_cat                    = len(ori_cardinalities) > 0
        self.cat_ori_embedding_layer    = CategoricalEmbeddings(ori_cardinalities, dim_model) if self.has_cat else None
        self.num_ori_embedding_layer    = ContinuousEmbeddings(n_features=num_features, d_embedding=dim_model)
        self.bte_embedding_layer        = ContinuousEmbeddings(n_features=statistic_features, d_embedding=dim_model)

        # === Positional Encoding ===
        total_tokens = num_features + cat_features
        seg_per_token = statistic_features // total_tokens
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, total_tokens, seg_per_token + 1, dim_model) # +1 : Original Data
        )

        # === Two-Stage Attention Encoder ===
        self.TwoStage = StackedTwoStageAttention(
            num_layers=num_layers_cross,
            d_model=dim_model,
            n_heads=num_heads,
            d_ff=dim_ff,
            attention_dropout=att_dropout,
            residual_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            return_attention=return_attention
        )

        # === Classifier ===
        self.classifier = nn.Linear(2 * dim_model, num_labels)

    
    def forward(self, cat_data, num_data, bte_data):
        
        # === cat/num lengths ===
        batch_size = bte_data.shape[0]
        cat_len = cat_data.shape[1] if cat_data is not None and cat_data.shape[1] > 0 else 0
        num_len = num_data.shape[1] if num_data is not None and num_data.shape[1] > 0 else 0
        ori_seq_len = num_len + cat_len

        # === CAT ===
        if  cat_len > 0:
            cat_emb = self.cat_ori_embedding_layer(cat_data).unsqueeze(2)
        else:
            cat_emb = torch.empty(batch_size, 0, 1, self.num_ori_embedding_layer.d_embedding, device=bte_data.device)
                     
        # === NUM ===
        if num_len > 0:
            num_emb = self.num_ori_embedding_layer(num_data).unsqueeze(2)
        else:
            num_emb = torch.empty(batch_size, 0, 1, self.num_ori_embedding_layer.d_embedding, device=bte_data.device)

        # === BTE ===
        bte_embedded    = self.bte_embedding_layer(bte_data)
        seg_per_token   = bte_embedded.shape[1] // ori_seq_len
        bte_embedded    = bte_embedded.view(batch_size, -1, seg_per_token, bte_embedded.shape[-1])
        
        # === Combine ===
        tmp_embedded = torch.cat([cat_emb, num_emb], dim=1)
        all_embedded = torch.cat([tmp_embedded, bte_embedded], dim=2)
        all_with_pos = all_embedded + self.pos_embeddings

        # === Two-Stage Attention ===
        final_out, _, _= self.TwoStage(all_with_pos)
        global_token = final_out[:, 0, :, :].reshape(batch_size, -1) 
        
        logits = self.classifier(global_token)

        return logits.squeeze(1)
