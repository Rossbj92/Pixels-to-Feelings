import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import time
import copy

def train_val_split(files,
                    target,
                    test_size,
                    random_state,
                    stratify=True):
    """

    """
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(files,
                                                          target,
                                                          test_size=test_size,
                                                          random_state=random_state,
                                                          stratify=target
                                                         )
    else:
        X_train, X_val, y_train, y_val = train_test_split(files,
                                                          target,
                                                          test_size=test_size
                                                         )
    train_split = pd.concat([X_train, y_train], axis = 1)
    val_split = pd.concat([X_val, y_val], axis = 1)
    return train_split, val_split

class ImgDataset(Dataset):
    """Create image dataset.

    Attributes:
      csv_file (string): Path to the csv file with annotations.
      root_dir (string): Directory with all the images.
      transform (callable, optional): Optional transform to be applied
        on a sample.
    """
    def __init__(self, df, root_dir, percent_sample=None, transform=None):
        """Inits class with df, root_dir, percent_sample, tarnsform"""
        self.img_df=df
        self.root_dir=root_dir
        self.transform=transform
        self.percent_sample=percent_sample

    def __len__(self):
        if self.percent_sample:
            assert self.percent_sample > 0.0, 'Percentage to sample must be >= 0 and <= 1.'
            assert self.percent_sample <= 1.0, 'Percentage to sample must be >= 0 and <= 1.'
            return int(np.floor(len(self.img_df) * self.percent_sample))
        else:
            return len(self.img_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_df.iloc[idx, 0])
        X = PIL.Image.open(img_name).convert('RGB') # Some images in greyscale, so converting to ensure
                                                    # 3 channels (1 causes issues in transformers)
        y = self.img_df.iloc[idx, 1]

        if self.transform:
            X = self.transform(X)

        return X, y


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Developed by: https://github.com/Bjarten/early-stopping-pytorch

    Attributes:
      patience (int): How long to wait after last time validation loss improved.
                    Default: 7
      verbose (bool): If True, prints a message for each validation loss improvement.
                    Default: False
      delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                    Default: 0
      path (str): Path for the checkpoint to be saved to.
                    Default: 'checkpoint.pt'
      trace_func (function): trace print function.
                    Default: print
    """
    def __init__(self, patience=7, delta=0, trace_func=print):
        """Inits class with patience, verbose, delta, and trace_func."""
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """

        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_model(path,
               epoch,
               model,
               optimizer,
               loss,
               acc):
    """

    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc': acc
                }, path)

def train_model(model,
                dataloader,
                criterion,
                optimizer,
                save_path_loss,
                save_path_acc,
                num_epochs=25):
    """

    """
    since = time.time()

    log_df = pd.DataFrame({'epoch':[],
                          'train_loss': [],
                          'val_loss': [],
                          'train_acc': [],
                          'val_acc': []})
    log_filename = f'{save_path_loss.split("_")[0]}_logs.csv'

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

#     best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=7, verbose=True)

    for epoch in range(num_epochs):

        epoch_time = time.time()

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

#         if early_stopping.early_stop:
#                     print("Early stopping")
#                     break

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_time_elapsed = time.time() - epoch_time
            print('Training complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch % 10 == 0:
                save_model(f'epoch_{epoch}.tar', epoch, model, optimizer, epoch_loss, epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print('saving model - best accuracy')
                save_model(save_path_acc, epoch, model, optimizer, epoch_loss, epoch_acc)
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print('saving model - best loss')
                save_model(save_path_loss, epoch, model, optimizer, epoch_loss, epoch_acc)
            # save metrics
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
#                 early_stopping(epoch_loss, model)

#                 if early_stopping.early_stop:
#                     print("Early stopping")
#                     break
            #early stopping
                log_df = log_df.append(pd.DataFrame({'epoch': [epoch],
                                                     'train_loss': [train_loss_history[-1]],
                                                     'val_loss': [val_loss_history[-1]],
                                                     'train_acc': [train_acc_history[-1]],
                                                     'val_acc': [val_acc_history[-1]]}))
                log_df.to_csv(log_filename)

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    checkpoint = torch.load(save_path_acc)
    print('Best val Acc: {:4f}'.format(checkpoint['acc']))
    print(f'Logs saved to {log_filename}')


    model.load_state_dict(checkpoint['model_state_dict'])

    return model
