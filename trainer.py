import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


class Trainer():
    def __init__(self, model, optimizer, loss_fn=nn.CrossEntropyLoss(), save_model_path='model_saved_dl/best_model.pth'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)  
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_model_path = save_model_path

        save_dir = os.path.dirname(self.save_model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def train_step(self, inputs, targets, attention_mask):
        self.model.train()

        inputs, targets, attention_mask = inputs.to(self.device), targets.to(self.device), attention_mask.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(inputs, attention_mask=attention_mask) 

        loss = self.loss_fn(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, inputs, targets, attention_mask):
        self.model.eval()

        inputs, targets, attention_mask = inputs.to(self.device), targets.to(self.device), attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs, attention_mask=attention_mask)  

            loss = self.loss_fn(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            accuracy = (preds == targets).float().mean().item()

        return loss.item(), accuracy


    def fit(self, train_loader, val_loader, epochs=10):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = 0.0
            val_loss = 0.0
            val_accuracy = 0.0

            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
            for inputs, targets, attention_mask in train_loop:  
                loss = self.train_step(inputs, targets, attention_mask)  
                train_loss += loss
                train_loop.set_postfix(train_loss=train_loss / len(train_loader))

            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)
            for inputs, targets, attention_mask in val_loop: 
                loss, accuracy = self.eval_step(inputs, targets, attention_mask)  
                val_loss += loss
                val_accuracy += accuracy
                val_loop.set_postfix(val_loss=val_loss / len(val_loader), val_accuracy=val_accuracy / len(val_loader))

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f"Best model saved with Val Loss: {best_val_loss:.4f} at {self.save_model_path}")

    def inference(self, data_loader):
        self.model.eval()

        predictions = []
        actual_values = []

        with torch.no_grad():  
            for inputs, targets, attention_mask in data_loader:  
                inputs, targets, attention_mask = inputs.to(self.device), targets.to(self.device), attention_mask.to(self.device)

                outputs = self.model(inputs, attention_mask=attention_mask)  

                predicted_classes = torch.argmax(outputs, dim=1).cpu()
                true_classes = targets.cpu()

                predictions.append(predicted_classes)
                actual_values.append(true_classes)

        predictions = torch.cat(predictions, dim=0).numpy()
        actual_values = torch.cat(actual_values, dim=0).numpy()

        return predictions, actual_values

    def evaluate_metrics(self, val_loader, model_path='model_saved_dl/best_model.pth'):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)  
        self.model.eval()  

        true_values = []
        predictions = []
        prediction_scores = []

        with torch.no_grad():  
            for inputs, targets, attention_mask in val_loader: 
                inputs, targets, attention_mask = inputs.to(self.device), targets.to(self.device), attention_mask.to(self.device)

                outputs = self.model(inputs, attention_mask=attention_mask) 

                predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()
                true_classes = targets.cpu().numpy()
                
                prediction_scores.extend(outputs[:, 1].cpu().numpy()) 

                true_values.extend(true_classes)
                predictions.extend(predicted_classes)

        true_values = np.array(true_values)
        predictions = np.array(predictions)
        prediction_scores = np.array(prediction_scores)

        accuracy = accuracy_score(true_values, predictions)
        precision = precision_score(true_values, predictions, average='weighted')
        recall = recall_score(true_values, predictions, average='weighted')
        f1 = f1_score(true_values, predictions, average='weighted')

        auroc = roc_auc_score(true_values, prediction_scores)

        conf_matrix = confusion_matrix(true_values, predictions)

        print(f"Classification Metrics\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nAUROC: {auroc:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
