from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy,MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pdb
import math


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, last_layer_dim_pi), nn.ReLU()
        )


        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
     
        return self.value_net(features)


class CustomActorCriticPolicy(MaskableMultiInputActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param cnn_output_dim: (int) Number of features extracted by the CNN.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        attention_d_model: int = 64, # Embedding dimension for attention
        attention_nhead: int = 4,     # Number of attention heads
        attention_dropout: float = 0.0 # Dropout for attention
    ):
        
        # --- CNN Feature Dimension (for bay_state) ---
        # This remains the same as before
        
        # --- Attention Feature Dimension (for containers) ---
        # After attention and aggregation (e.g., mean pooling), the output dim will be attention_d_model
        attention_output_dim = attention_d_model

        top_weights_dim = observation_space["top_weights"].shape[0]
        mlp_output_dim_weights = 256 
        attention_topWeight_dim = attention_d_model

        # --- Calculate Total Feature Dimension ---
        total_features_dim = attention_topWeight_dim + cnn_output_dim
        
        super().__init__(observation_space, features_dim=total_features_dim)

        # Ensure attention_d_model is divisible by attention_nhead
        if attention_d_model % attention_nhead != 0:
            raise ValueError(f"attention_d_model ({attention_d_model}) must be divisible by "
                             f"attention_nhead ({attention_nhead})")

        # --- CNN for 'bay_state' (same as before) ---
        n_input_channels = 1 # Assuming bay_state is single channel HxW
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate CNN flattened dimension dynamically
        with th.no_grad():
            dummy_input_shape = (1, n_input_channels) + observation_space["bay_state"].shape
            dummy_input = th.zeros(dummy_input_shape)
            cnn_flatten_dim = self.cnn(dummy_input).shape[-1]
            
        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_flatten_dim, cnn_output_dim),
            nn.ReLU()
        )

        # --- MLP for 'top_weights' (same as before) ---
        # self.mlp_top_weights = nn.Sequential(
        #     nn.Linear(top_weights_dim, 256), # 输入维度是 cont_weights 的长度
        #     nn.ReLU(),
        #     nn.Linear(256, mlp_output_dim_weights),
        #     nn.ReLU()
        # )
        # --- Attention for 'top_weights' ---
        # self.top_weights_input_proj = nn.Linear(1, attention_d_model)
        # self.attention_topWeight = nn.MultiheadAttention(
        #     embed_dim=attention_d_model, 
        #     num_heads=attention_nhead, 
        #     dropout=attention_dropout,
        #     batch_first=True # IMPORTANT: Input/Output shape is (Batch, Seq, Feature)
        # )


        # --- Self-Attention for 'cont_weights' and 'cont_port' ---
        
        # 1. Combine 'cont_weights' and 'cont_port' features. 
        #    Each container will have 2 features.
        container_feature_dim = 2
        
        # 2. Input projection layer: Map combined features to attention_d_model
        self.container_input_proj = nn.Linear(container_feature_dim, attention_d_model)
        
        # 3. Multi-Head Self-Attention Layer
        #    batch_first=True expects input shape (Batch, Sequence Length, Embedding Dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_d_model, 
            num_heads=attention_nhead, 
            dropout=attention_dropout,
            batch_first=True # IMPORTANT: Input/Output shape is (Batch, Seq, Feature)
        )

        # 4. Positional Encoding (Optional)
        # self.max_cont_num = observation_space["cont_weights"].shape[0] 
        # pe = th.zeros(self.max_cont_num, attention_d_model)
        # position = th.arange(0, self.max_cont_num, dtype=th.float).unsqueeze(1) # Shape (max_cont_num, 1)
        # div_term = th.exp(th.arange(0, attention_d_model, 2).float() * (-math.log(10000.0) / attention_d_model)) # Shape (d_model/2)
    
        # pe[:, 0::2] = th.sin(position * div_term)
        # pe[:, 1::2] = th.cos(position * div_term)
        # self.register_buffer('pe', pe) 
        
        # (Optional) Add Layer Normalization and Feed-Forward after attention (like in Transformers)
        # self.layer_norm1 = nn.LayerNorm(attention_d_model)
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(attention_d_model, attention_d_model * 2),
        #     nn.ReLU(),
        #     nn.Linear(attention_d_model * 2, attention_d_model)
        # )
        # self.layer_norm2 = nn.LayerNorm(attention_d_model)
        # self.dropout = nn.Dropout(attention_dropout) # If adding FF layer

    def forward(self, observations: TensorDict) -> th.Tensor:
        # --- Process 'bay_state' with CNN ---
        bay_state_obs = observations["bay_state"].float().unsqueeze(1) # (B, 1, H, W)
        cnn_latent = self.cnn(bay_state_obs)
        cnn_features = self.cnn_linear(cnn_latent) # (B, cnn_output_dim)
        top_weights_features = cnn_features

        # --- Process 'top_weights' with MLP ---
        # top_weights_obs = observations["top_weights"].float()
        # top_weights_features = self.mlp_top_weights(top_weights_obs) # (B, mlp_output_dim_weights)

        # --- Process 'top_weights' with Attention ---
        # top_weights_obs = observations["top_weights"].float().unsqueeze(-1) # (B, top_weights_dim, 1)
        # # 1. Project features: (B, top_weights_dim, 1) -> (B, top_weights_dim, attention_d_model)
        # projected_top_weights = self.top_weights_input_proj(top_weights_obs) 
        
        # # 2. Apply Self-Attention:
        # top_attn_output, _ = self.attention_topWeight(
        #     query=projected_top_weights, 
        #     key=projected_top_weights, 
        #     value=projected_top_weights
        # )
        # top_weights_features = top_attn_output.mean(dim=1) # (B, attention_d_model)
        

        # --- Process 'cont_weights' and 'cont_port' with Attention ---
        cont_weights_obs = observations["cont_weights"].float() # (B, cont_num)
        #cont_port_obs = observations["cont_port"].float()       # (B, cont_num)
        
        # 1. Combine features: (B, cont_num) + (B, cont_num) -> (B, cont_num, 2)
        #    We stack them along the last dimension.
        combined_cont_features = th.stack((cont_weights_obs, cont_weights_obs), dim=-1)
        
        
        # 2. Project features: (B, cont_num, 2) -> (B, cont_num, attention_d_model)
        projected_cont_features = self.container_input_proj(combined_cont_features)
        
        # --- Optional: Add Positional Encoding if order matters ---
        # seq_len = projected_cont_features.size(1)
        # pos_encoding = self.pe[:seq_len, :].unsqueeze(0) 
        # projected_cont_features = projected_cont_features + pos_encoding

        # 3. Apply Self-Attention:
        #    Query, Key, Value are all the same for self-attention
        #    Input: (B, cont_num, attention_d_model)
        #    Output: (attn_output, attn_weights)
        #       attn_output shape: (B, cont_num, attention_d_model)
        attn_output, _ = self.attention(
            query=projected_cont_features, 
            key=projected_cont_features, 
            value=projected_cont_features,
            need_weights=False # We usually don't need the weights themselves for feature extraction
        )
        
        # --- Optional: Apply LayerNorm and FeedForward (Transformer Block style) ---
        # attn_output = projected_cont_features + self.dropout(attn_output) # Residual connection + Dropout
        # attn_output = self.layer_norm1(attn_output)
        # ff_output = self.feed_forward(attn_output)
        # attn_output = attn_output + self.dropout(ff_output) # Residual connection + Dropout
        # attn_output = self.layer_norm2(attn_output)
        # ------------------------------------------------------------------------

        # 4. Aggregate Attention Output:
        #    Pool the features across the 'cont_num' dimension.
        #    Using mean pooling here: (B, cont_num, attention_d_model) -> (B, attention_d_model)
        #    You could also use max pooling: attn_output.max(dim=1).values
        aggregated_attention_features = attn_output.max(dim=1).values  

       
        # --- Concatenate CNN features and Aggregated Attention features ---
        combined_features = th.cat((top_weights_features, aggregated_attention_features), dim=1)
        # Expected shape: (B, cnn_output_dim + attention_d_model) 
        # which should match self.features_dim
        
        return combined_features

