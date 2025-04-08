from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from layers.emb_siddhi import Embeddings
from models.siddhi import MatrixFactorizationLayer,CrossAttentionLayer,WeightedFusionLayer,UnifiedModel
from data_provider.data_factory import data_provider
class Exp_Classification_Siddhi():
    def __init__(self,args):
        super(Exp_Classification_Siddhi,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args=args
        self.input_dim_p = 65
        self.input_dim_c=19
        self.latent_dim = 128
        self.batch_size_patch = 32
        self.batch_size_channel = 32
        self.epochs = 4000
        self.learning_rate = 0.001
        self.model = UnifiedModel(input_dim_p=self.input_dim_p,input_dim_c=self.input_dim_c, latent_dim=self.latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
    def _get_data(self, flag):
        # random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    def _load_train_data(self):
        train_data, train_loader = self._get_data(flag="TRAIN")
        # vali_data, vali_loader = self._get_data(flag="VAL")
        # test_data, test_loader = self._get_data(flag="TEST")
        return train_data,train_loader
    def _load_vali_data(self):
        vali_data, vali_loader = self._get_data(flag="VAL")
        return vali_data,vali_loader
    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self.criterion
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        emb = Embeddings(self.args)
        preds = []
        trues = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label, padding) in enumerate(vali_loader):
                X_patch_batch, X_channel_batch = emb(batch_x)
                X_patch_batch = X_patch_batch[0]
                X_channel_batch = X_channel_batch[0]

                self.optimizer.zero_grad()
                outputs = self.model(X_patch_batch, X_channel_batch)
                loss = criterion(outputs.squeeze().float(), label.squeeze().float())
                total_loss.append(loss.item())
                preds.append(outputs)
                trues.append(label)

        total_loss = sum(total_loss) / len(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        probs = torch.sigmoid(preds)  # For binary classification
        predictions = (probs > 0.5).int()

        # Detach and convert to numpy (only here, no .cpu())
        predictions = predictions.numpy()
        probs = probs.numpy()
        trues = trues.flatten().numpy()

        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
        }

        self.model.train()
        return total_loss, metrics_dict


    def train(self,setting):
        model=self.model
        emb=Embeddings(self.args)
        criterion=self.criterion
        optimizer=self.optimizer
        train_data,train_loader=self._load_train_data()
        vali_data,vali_loader=self._load_vali_data()
        for epoch in range(self.epochs):
            model.train()  # Set model to training mode
            train_loss = 0.0
            correct = 0
            total = 0
    
            for batch_idx, (X_batch, y_batch,padding) in enumerate(train_loader):
                # Forward pass
                X_patch_batch,X_channel_batch=emb(X_batch)
                X_patch_batch=X_patch_batch[0]
                X_channel_batch=X_channel_batch[0]
                optimizer.zero_grad()  # Zero gradients before backward pass
                output = model(X_patch_batch, X_channel_batch)
                # Compute loss
                loss = criterion(output.squeeze().float(), y_batch.squeeze().float())  # Squeeze to match shapes
                loss.backward()  # Backpropagation
                optimizer.step()  # Optimizer step
        
                # Track loss and accuracy
                train_loss += loss.item()
                predicted = (output > 0.5).float()  # Binary classification (0 or 1)
                correct += (predicted.squeeze() == y_batch.squeeze()).sum().item()
                total += y_batch.size(0)
    
        # Validation loop
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct_val = 0
            total_val = 0
    
            with torch.no_grad():  # No gradients needed for validation
                for X_batch, y_batch,padding in vali_loader:
                    X_patch_batch,X_channel_batch=emb(X_batch)
                    X_patch_batch=X_patch_batch[0]
                    X_channel_batch=X_channel_batch[0]
                    output = model(X_patch_batch, X_channel_batch)
                    loss = criterion(output.squeeze().float(), y_batch.squeeze().float())
            
                    val_loss += loss.item()
                    predicted = (output > 0.5).float()
                    correct_val += (predicted.squeeze() == y_batch.squeeze()).sum().item()
                    total_val += y_batch.size(0)
    
    # Print training and validation statistics for each epoch
        print(f'Epoch {epoch+1}/{self.epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {100 * correct/total:.2f}%')
        print(f'Validation Loss: {val_loss/len(vali_loader):.4f}, Validation Accuracy: {100 * correct_val/total_val:.2f}%')

    # Save the model after training (optional)
        torch.save(model.state_dict(), 'unified_model.pth')


