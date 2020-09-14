
# Data manipulation
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Modeling
import PIL.Image
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

def load_data():
    """Loads train/test CSVs."""
    train = pd.read_csv('../data/csvs/train.csv')
    test = pd.read_csv('../data/csvs/test.csv')
    return train, test

def train_val_split(X,
                    y,
                    test_size,
                    random_state,
                    stratify=True):
    """Splits data into training and validation sets.

    Args:
        X (series): Pandas series containing image names.
        Y (series): Pandas series containing image labels.
        test_size (float): Percentage of data used for
          validation sample.
        random_state (int): Seed.
        stratify (bool): Whether to stratify split based on
          label distributions. Default is True.

    Returns:
        Train and validation dataframes.
    """
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          test_size=test_size,
                                                          random_state=random_state,
                                                          stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          test_size=test_size,
                                                          random_state=random_state)
    train_split = pd.concat([X_train, y_train], axis = 1)
    val_split = pd.concat([X_val, y_val], axis = 1)
    return train_split, val_split

class ImgDataset(Dataset):
    """Creates image dataset using Pytorch Dataset class.

    Attributes:
        df (dataframe): Pandas dataframe containing image file
          names in 1st column and labels in 2nd column.
        root_dir (str): Directory with all the images.
        percent_sample (float): Percentage of data to use for
          modeling. If None, all data are used -- default is None.
        transform (callable): Optional transform to be applied on
          a sample.
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
      patience (int): How long to wait after last time validation
        loss improved. Default: 7
      delta (float): Minimum change in the monitored quantity to
        qualify as an improvement. Default: 0
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

def weighted_sample(df, col):
    """Calculates class weights for use in Pytorch's WeightedRandomSampler.

    Weights for each class are computed using their inverse frequency. That is,
    a class that appears 100 times will have a weight of 1/100, whereas a class
    that appears 10 times will have a weight of 1/10.

    Args:
        df (dataframe): Pandas dataframe containing images and labels.
        col (str): String corresponding to label column.

    Returns:
        List containing weights for each class.
    """
    class_sample_counts = df[col].value_counts().reset_index().sort_values('index')[col].tolist()
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[df[col].tolist()]
    return samples_weights

def save_model(path,
               epoch,
               model,
               optimizer,
               loss,
               acc):
    """Saves Pytorch model.

    Args:
        Path (str): Path to save model in.
        Epoch (int): Current epoch in model training.
        Model (obj): Pytorch model.
        Optimizer (obj): Pytorch optimizer used for 'model' arg.
        Loss, acc (float): Current epoch loss/accuracy value.

    Returns:
        A dictionary saved at the specified path with 'epoch',
        'model_state_dict', 'optimizer_state_dict', 'loss', and
        'acc' as keys, and the args as values.
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc': acc
                }, path)
    return



def train_model(model,
                dataloader,
                criterion,
                optimizer,
                save_path,
                num_epochs=25,
                scheduler=None,
                early_stopping=None):
    """Trains a Pytorch model.

    Args:
        Model (obj): Pytorch model.
        Dataloader (dict): A dictionary with 'train' and 'val'
          as keys and Pytorch dataloader classes as values.
        Criterion (obj): A Pytorch loss function.
        Optimizer (obj): A Pytorch optimizer.
        Save_path (str): Path to save model checkpoint.
        Num_epochs (int): Number of epochs to train. Default: 25.
        Scheduler (obj): A Pytorch learning rate scheduler. Default: None.
        Early_stopping (obj): An instance of the 'EarlyStopping' class.
          Default: None.

    Returns:
        The model with the lowest validation loss across all epochs.
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

    best_acc = 0.0
    best_loss = np.inf

    if early_stopping:
        early_stopping = early_stopping

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if early_stopping.early_stop:
                    print("Early stopping")
                    break

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

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print('saving model - best loss')
                save_model(save_path, epoch, model, optimizer, epoch_loss, epoch_acc)
            # save metrics
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

                scheduler.step(epoch_loss)

                if early_stopping:
                    early_stopping(epoch_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
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

    checkpoint = torch.load(save_path)
    print('Best val Acc: {:4f}'.format(checkpoint['acc']))
    print(f'Logs saved to {log_filename}')


    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_model(path, base, old_classes, new_classes, device):
    """Loads state dictionary of Pytorch model.

    Args:
      Path (str): Path where model is saved.
      Base (obj): Pytorch model class matching parameters of
        loaded state dictionary.
      Old_classes (int): Number of outputs in fully-connected
        layer of 'base' arg.
      New_classes (int): Number of outputs for fully-connected
        layer of new model.
      Device (obj): Pytorch device.

    Returns:
      Loaded Pytorch model.
    """
    model = base
    num_ftrs = model_ft.fc.in_features
    model.fc = m.nn.Linear(num_ftrs, old_classes)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.fc = m.nn.Linear(num_ftrs, new_classes)
    model = model.to(device)
    return model

def evaluate_model(model, dataloader):
    """Evaluates model on specified data.

    Before calling this function, ensure that the model is set
    to 'model.eval()'. Metric that is evaluated is accuracy.

    Args:
      Model (obj): Pytorch model.
      Dataloder (obj): Pytorch dataloader class holding data
        for model to predict.

    Returns:
        Accuracy score for model on dataloader data.
    """
    running_corrects = 0

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / len(dataloader.dataset)

    return acc

def true_and_preds(model, dataloader, batch_size, device):
    """Returns true and predicted values of model.

    Before calling this function, ensure that the model is set
    to 'model.eval()'.

    Args:
        Model (obj): Pytorch model.
        Dataloder (obj): Pytorch dataloader class holding data
          for model to predict.
        Batch_size (int): Mini-batch size.
        Device (obj): Pytorch device.

    Returns:
        2 arrays: actual and predicted class labels.
    """
    y_true = np.zeros(len(dataloader.dataset))
    y_pred = np.zeros(len(dataloader.dataset))

    for batch, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

        if len(outputs) == batch_size:
            y_true[batch*batch_size:(batch+1)*batch_size] = labels.cpu().numpy()
            y_pred[batch*batch_size:(batch+1)*batch_size] = preds.cpu().numpy()
        else:
            y_true[-len(outputs):] = labels.cpu().numpy()
            y_pred[-len(outputs):] = preds.cpu().numpy()
    return y_true, y_pred
