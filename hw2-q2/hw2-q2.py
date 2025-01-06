#!/usr/bin/env python

# Deep Learning Homework 2

import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

import utils


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=None,
            maxpool=True,
            batch_norm=True,
            dropout=0.0
        ):
        super().__init__()
        
        # Conv layer with kernel_size=3, stride=1, padding=1
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Batch normalization layer (or Identity if batch_norm=False)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # MaxPool layer with kernel_size=2x2 and stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Updated order: Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x

class CNN(nn.Module):
    def __init__(self, dropout_prob, maxpool=True, batch_norm=True, conv_bias=True):
        super(CNN, self).__init__()
        channels = [3, 32, 64, 128]
        fc1_out_dim = 1024
        fc2_out_dim = 512
        self.maxpool = maxpool
        self.batch_norm = batch_norm

        # Initialize three convolutional blocks
        self.conv_block1 = ConvBlock(channels[0], channels[1], kernel_size=3, 
                                   maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob)
        self.conv_block2 = ConvBlock(channels[1], channels[2], kernel_size=3, 
                                   maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob)
        self.conv_block3 = ConvBlock(channels[2], channels[3], kernel_size=3, 
                                   maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP block with batch normalization
        self.fc1 = nn.Linear(channels[3], fc1_out_dim)  # Input is now just the number of channels
        self.bn1 = nn.BatchNorm1d(fc1_out_dim) if batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.bn2 = nn.BatchNorm1d(fc2_out_dim) if batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(fc2_out_dim, 6)  # 6 classes for Intel dataset

    def forward(self, x):
        # Input reshaping
        x = x.reshape(x.shape[0], 3, 48, -1)
        
        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten after global pooling
        
        # MLP block with batch norm
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
    
    
def get_number_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels
  
def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')



def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])

def plot_metrics(epochs, metrics, ylabel='', name=''):
    """
    Plot metrics using plotly and save as PDF
    
    Args:
        epochs: list of epoch numbers
        metrics: list of metric values to plot
        ylabel: label for y-axis
        name: filename to save the plot
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=metrics,
        mode='lines',
        name=ylabel
    ))
    
    fig.update_layout(
        title=f'{ylabel} vs Epochs',
        xaxis_title='Epoch',
        yaxis_title=ylabel,
        template='plotly_white',
        showlegend=True
    )
    
    # Save as PNG
    fig.write_image(f'{name}.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int,
                       help="Number of epochs to train for.")
    parser.add_argument('-batch_size', default=8, type=int,
                       help="Size of training batch.")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_batch_norm', action='store_true')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz')
    parser.add_argument('-device', choices=['cpu', 'cuda', 'mps'], default='mps')
    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils.configure_seed(seed=42)

    # Learning rates to test
    learning_rates = [0.01]
    results = {}

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Load data
        data = utils.load_dataset(data_path=opt.data_path)
        dataset = utils.ClassificationDataset(data)
        train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
        dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)
        test_X, test_y = dataset.test_X.to(opt.device), dataset.test_y.to(opt.device)

        # Initialize model
        model = CNN(opt.dropout, maxpool=not opt.no_maxpool, 
           batch_norm=not opt.no_batch_norm).to(opt.device)

        # Initialize optimizer with current learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=opt.l2_decay)
        criterion = nn.NLLLoss()

        # Training loop
        epochs = np.arange(1, opt.epochs + 1)
        train_mean_losses = []
        valid_accs = []
        train_losses = []

        for ii in epochs:
            print(f'\nTraining epoch {ii}')
            model.train()
            epoch_losses = []
            
            for X_batch, y_batch in train_dataloader:
                X_batch = X_batch.to(opt.device)
                y_batch = y_batch.to(opt.device)
                loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
                epoch_losses.append(loss)

            mean_loss = torch.tensor(epoch_losses).mean().item()
            train_mean_losses.append(mean_loss)
            print(f'Training loss: {mean_loss:.4f}')

            val_acc, val_loss = evaluate(model, dev_X, dev_y, criterion)
            valid_accs.append(val_acc)
            print(f"Valid loss: {val_loss:.4f}")
            print(f'Valid acc: {val_acc:.4f}')

        # Final test accuracy
        test_acc, _ = evaluate(model, test_X, test_y, criterion)
        print(f'Final Test acc with lr={lr}: {test_acc:.4f}')
        
        # Store results for this learning rate
        results[lr] = {
            'train_losses': train_mean_losses,
            'valid_accs': valid_accs,
            'test_acc': test_acc
        }

        # Save individual plots
        sufix = f"lr_{lr}"
        plot_metrics(epochs, train_mean_losses, ylabel='Loss', 
                    name=f'CNN-3-train-loss-with-batchnorm')
        plot_metrics(epochs, valid_accs, ylabel='Accuracy', 
                    name=f'CNN-3-valid-accuracy-with-batchnorm')

    # Create combined plots using plotly
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Training Loss', 'Validation Accuracy'))

    # Add traces for each learning rate
    colors = ['blue', 'red', 'green']
    for i, lr in enumerate(learning_rates):
        epochs = list(range(1, opt.epochs + 1))
        
        # Training loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=results[lr]['train_losses'], 
                      name=f'lr={lr}', line=dict(color=colors[i])),
            row=1, col=1
        )
        
        # Validation accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=results[lr]['valid_accs'], 
                      name=f'lr={lr}', line=dict(color=colors[i])),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(height=500, width=1000, title_text="Training Metrics")
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # Save plots
    fig.write_image("training_results.png")

    # Print best learning rate
    best_lr = max(results.keys(), 
                 key=lambda lr: max(results[lr]['valid_accs']))
    print(f"\nBest learning rate: {best_lr}")
    print(f"Best test accuracy: {results[best_lr]['test_acc']:.4f}")
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()