import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

import pdb

class PositionEmbedding(nn.Module):
    def __init__(self, args):
        super(PositionEmbedding, self).__init__()
        self.dim = args.dim
        self.pos_embed = nn.Embedding(num_embeddings=args.max_len, embedding_dim=self.dim)
        self.Drop = nn.Dropout(args.dropout)

    def forward(self, seq_h):
        position_ids = torch.arange(0, seq_h.size(1), dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_h.size(1))
        pe = self.pos_embed(position_ids).repeat(seq_h.size(0), 1, 1)
        return self.Drop(pe)


class TrmEncoder(nn.Module):
    """
    Implementation of Transformer: Cross-Attention
    """

    def __init__(self, input_size, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1):
        super(TrmEncoder, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        
        #
        self.port_embed = nn.Embedding(num_embeddings=4, embedding_dim=self.d_k)
        self.weight_embed = nn.Linear(1, self.d_k)

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V ,attn_mask, attn_bias ,epsilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + epsilon)

        pad_mask = attn_mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
        attn_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool()
        mask_ = attn_mask + pad_mask
        Q_K = Q_K.masked_fill(mask_, -2 ** 32 + 1) 
        
        if attn_bias is not None:
            attn_bias = attn_bias.masked_fill(mask_, -2 ** 32 + 1)
            attn_weight = F.softmax(Q_K, dim=-1) + F.softmax(attn_bias, dim=-1)  #N*N

        else:
            attn_weight = F.softmax(Q_K, dim=-1)

        attn_weight = self.dropout(attn_weight)
        return attn_weight @ V

    def multi_head_attention(self, Q, K, V, mask, bias):

        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, k_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, v_len, self.d_v)

        mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
        mask = mask.reshape(-1, mask.size(-1))
        if bias is not None:
            bias = bias.unsqueeze(dim=1).expand(-1, self.n_heads, -1, -1)
            bias = bias.reshape(-1, mask.size(-1), mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask, bias)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, weights, unload_port, bias=None):
        fea_weights = torch.tensor(weights, dtype=torch.float32)
        fea_unload_port = torch.tensor(unload_port, dtype=torch.int32)

        """[batch, bay_width*bay_height, d_k]"""
        fea_unload_port = self.port_embed(fea_unload_port.view(-1)).view(1,-1,self.d_k) 
        fea_weights = self.weight_embed(fea_weights.view(-1,1)).view(1,-1,self.d_k) 
        features = fea_unload_port + fea_weights

        Q = features
        K = features
        V = features

        mask = torch.zeros(features.size(0), features.size(1),dtype=torch.bool)

        V_att = self.multi_head_attention(Q, K, V, mask, bias)
        

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(TransformerFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # 自己定义的
        #self.trm_encoder = TrmEncoder(input_size=128, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1)

        #
        self.embedding_dim = 16  
        self.fc1 = nn.Linear(observation_space.shape[1], self.embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=2, dim_feedforward=64, batch_first=True),
            num_layers=2
        )
        self.fc2 = nn.Linear(self.embedding_dim, self._features_dim)  # 映射到 PPO 需要的维度


    def forward(self, observation): #[batch, 2, bay_width*bay_height]
        
       
        weights = observation[:,0]
        unload_port = observation[:,1]
        x = self.fc1(weights)
        x = self.transformer(x)
        features = self.fc2(x)
        #pdb.set_trace()
        
        #features = self.trm_encoder(weights, unload_port) # [batch, bay_width*bay_height, d_k]
        #features = features.mean(dim=1) 

        #
        # features = observation
        # features = features.view(features.size(0), -1) 
        # features = nn.Linear(features.size(1), self._features_dim)(features)
        #print(features.size())
      
       
        
        
        return features

