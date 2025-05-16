import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class MatrixFactorizationLayer(nn.Module):
    def __init__(self, input_dim_p, input_dim_c, latent_dim):
        super(MatrixFactorizationLayer, self).__init__()
        self.W_patch = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.W_channel = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W_patch)
        nn.init.xavier_uniform_(self.W_channel)

    def forward(self, X_patch, X_channel):
        Z_patch = X_patch @ self.W_patch
        Z_channel = X_channel @ self.W_channel
        H_factor = F.relu(Z_patch @ Z_channel.transpose(1, 2))
        return H_factor



class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim_p,input_dim_c,latent_dim):
        super(CrossAttentionLayer, self).__init__()
        self.latent_dim=latent_dim
    def top_k_sparse_mask(self,scores,k):
      mean = scores.mean(dim=-1, keepdim=True)
      std = scores.std(dim=-1, keepdim=True)
      threshold = mean + std/8  

      mask = torch.where(scores >= threshold, torch.tensor(0.0, device=scores.device), torch.tensor(float('-inf'), device=scores.device))
      # print(mask)
      return mask
    def forward(self, X_patch, X_channel):
        # Patch to Channel
        A_patch_to_channel_attention_scores=X_patch @ X_channel.transpose(1, 2) / (self.latent_dim**0.5)
        A_patch_to_channel_attention_mask=self.top_k_sparse_mask(A_patch_to_channel_attention_scores,k=10)
        A_patch_to_channel_attention_scores+=A_patch_to_channel_attention_mask
        A_patch_to_channel = F.softmax(A_patch_to_channel_attention_scores, dim=-1)
        H_patch_to_channel = A_patch_to_channel @ X_channel
        
        # Channel to Patch
        A_channel_to_patch_attention_scores=X_channel @ X_patch.transpose(1, 2) / (self.latent_dim**0.5)
        A_channel_to_patch_attention_mask=self.top_k_sparse_mask(A_channel_to_patch_attention_scores,k=10)
        A_channel_to_patch_attention_scores+=A_channel_to_patch_attention_mask
        A_channel_to_patch = F.softmax(A_channel_to_patch_attention_scores, dim=-1)
        H_channel_to_patch = A_channel_to_patch @ X_patch
        
        # Hybrid representation
        H_hybrid = H_patch_to_channel @ H_channel_to_patch.transpose(1, 2)
        return H_hybrid


class WeightedFusionLayer(nn.Module):
    def __init__(self, input_dim_p, input_dim_c):
        super(WeightedFusionLayer, self).__init__()
        self.W_alpha = nn.Parameter(torch.empty(input_dim_c, input_dim_c))
        self.W_beta = nn.Parameter(torch.empty(input_dim_c, input_dim_c))
        self.W_out = nn.Parameter(torch.empty(49, 19))
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, H_factor, H_hybrid):
        alpha = F.softmax(H_factor @ self.W_alpha, dim=-1)
        beta = F.softmax(H_hybrid @ self.W_beta, dim=-1)
        H_final = alpha * H_factor + beta * H_hybrid
        u = torch.sum(H_final * self.W_out, dim=(1, 2)) + self.bias
        y = torch.sigmoid(u)
       
        return y

class UnifiedModel(nn.Module):
    def __init__(self, input_dim_p,input_dim_c, latent_dim):
        super(UnifiedModel, self).__init__()
        self.matrix_factorization = MatrixFactorizationLayer(input_dim_p,input_dim_c, latent_dim)
        self.cross_attention = CrossAttentionLayer(input_dim_p,input_dim_c,latent_dim)
        self.weighted_fusion = WeightedFusionLayer(input_dim_p,input_dim_c)
    
    def forward(self, X_patch, X_channel):
        # Matrix Factorization
        H_factor = self.matrix_factorization(X_patch, X_channel)
        
        # Cross Attention
        H_hybrid = self.cross_attention(X_patch, X_channel)
        
        # Weighted Fusion and Task-Specific Output

        y = self.weighted_fusion(H_factor, H_hybrid)
        
        return y



