import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from implementation import UnifiedModel

# Hyperparameters
input_dim_p = 65
input_dim_c=19
latent_dim = 128
batch_size_patch = 16
batch_size_channel = 16
epochs = 10
learning_rate = 0.001



class MatrixFactorizationLayer(nn.Module):
    def __init__(self, input_dim_p,input_dim_c, latent_dim):
        super(MatrixFactorizationLayer, self).__init__()
        self.W_patch = nn.Parameter(torch.randn(latent_dim, latent_dim)) # dxl
        self.W_channel = nn.Parameter(torch.randn(latent_dim, latent_dim)) # dxl
    
    def forward(self, X_patch, X_channel):
        # Project to shared latent spaces
        Z_patch = X_patch @ self.W_patch # pxd @ dxl = pxl
        Z_channel = X_channel @ self.W_channel # c
        # ReLU activation
        H_factor = F.relu(Z_patch @ Z_channel.mT)
        return H_factor


class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim_p,input_dim_c,latent_dim):
        super(CrossAttentionLayer, self).__init__()
    
    def forward(self, X_patch, X_channel):
        # Patch to Channel
        A_patch_to_channel = F.softmax(X_patch @ X_channel.mT / (latent_dim**0.5), dim=-1)
        H_patch_to_channel = A_patch_to_channel @ X_channel
        
        # Channel to Patch
        A_channel_to_patch = F.softmax(X_channel @ X_patch.mT / (latent_dim**0.5), dim=-1)
        H_channel_to_patch = A_channel_to_patch @ X_patch
        
        # Hybrid representation
        H_hybrid = H_patch_to_channel @ H_channel_to_patch.mT
        return H_hybrid


class WeightedFusionLayer(nn.Module):
    def __init__(self, input_dim_p,input_dim_c):
        super(WeightedFusionLayer, self).__init__()
        self.W_alpha = nn.Parameter(torch.randn(input_dim_c, input_dim_c))
        self.W_beta = nn.Parameter(torch.randn(input_dim_c, input_dim_c))
        self.W_out = nn.Parameter(torch.randn(49, 19))  # Change to output 1 value
        self.bias = nn.Parameter(torch.zeros(1))  # Bias term for binary output
    
    def forward(self, H_factor, H_hybrid):
        batch_size=H_factor.shape[0]


       

        alpha = F.softmax(H_factor @ self.W_alpha, dim=-1)
        beta = F.softmax(H_hybrid @ self.W_beta, dim=-1)
       
        # Weighted fusion
        H_final = torch.add(alpha*H_factor,beta*H_hybrid)
        
        # Output with a sigmoid activation for binary classification
        print(f'H_final {H_final.shape}')
        u=torch.sum(H_final * self.W_out,dim=(1,2)) + self.bias
        y = torch.sigmoid(u)
        print(y)
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



# Generate synthetic data for training (for demonstration purposes)
# X_patch = torch.randn(batch_size_patch,input_dim_p ,latent_dim)  # 500 patches, each of size input_dim
# X_channel = torch.randn(batch_size_channel,input_dim_c,latent_dim)  # 500 channels, each of size input_dim
# y = torch.randint(0, 2, (16, 1)).float()  # Binary labels
# # Split into train and validation sets
# X_patch_train, X_patch_val, X_channel_train, X_channel_val, y_train, y_val = train_test_split(X_patch, X_channel, y, test_size=0.2)

# # Create DataLoader for batching
# train_dataset = TensorDataset(X_patch_train, X_channel_train, y_train)
# val_dataset = TensorDataset(X_patch_val, X_channel_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=batch_size_patch, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size_patch, shuffle=False)

# # Model initialization
# model = UnifiedModel(input_dim_p=input_dim_p,input_dim_c=input_dim_c, latent_dim=latent_dim)

# # Optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification

# # Training loop
# for epoch in range(epochs):
#     model.train()  # Set model to training mode
#     train_loss = 0.0
#     correct = 0
#     total = 0
    
#     for batch_idx, (X_patch_batch, X_channel_batch, y_batch) in enumerate(train_loader):
#         # Forward pass
#         optimizer.zero_grad()  # Zero gradients before backward pass
#         output = model(X_patch_batch, X_channel_batch)
#         # Compute loss
#         loss = criterion(output.squeeze(), y_batch.squeeze())  # Squeeze to match shapes
#         loss.backward()  # Backpropagation
#         optimizer.step()  # Optimizer step
        
#         # Track loss and accuracy
#         train_loss += loss.item()
#         predicted = (output > 0.5).float()  # Binary classification (0 or 1)
#         correct += (predicted.squeeze() == y_batch.squeeze()).sum().item()
#         total += y_batch.size(0)
    
#     # Validation loop
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     correct_val = 0
#     total_val = 0
    
#     with torch.no_grad():  # No gradients needed for validation
#         for X_patch_batch, X_channel_batch, y_batch in val_loader:
#             output = model(X_patch_batch, X_channel_batch)
#             loss = criterion(output.squeeze(), y_batch.squeeze())
            
#             val_loss += loss.item()
#             predicted = (output > 0.5).float()
#             correct_val += (predicted.squeeze() == y_batch.squeeze()).sum().item()
#             total_val += y_batch.size(0)
    
#     # Print training and validation statistics for each epoch
#     print(f'Epoch {epoch+1}/{epochs}')
#     print(f'Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {100 * correct/total:.2f}%')
#     print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100 * correct_val/total_val:.2f}%')

# # Save the model after training (optional)
# torch.save(model.state_dict(), 'unified_model.pth')
