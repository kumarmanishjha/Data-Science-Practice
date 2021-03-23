import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from classifier import Net
from clf_dataset import ClassifierDataset


BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASS = 3 #Below code does not assume prior knowledge of number of classes 

def prediction_acc(y_pred, y_true):
    '''Returns prediction accuracy for a batch of output'''
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    

    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = acc * 100

    return acc


def train(model, train_loader, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Begin training.")
    for e in tqdm(range(1, EPOCHS+1)):
        # TRAINING with train data
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for i, data in enumerate(train_loader):
            X_train_batch, y_train_batch = data
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            labels = torch.max(y_train_batch, 1)[1]
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, labels)
            train_acc = prediction_acc(y_train_pred, labels)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()


        # Validation with val split  
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in test_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                labels = torch.max(y_val_batch, 1)[1]

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, labels)
                val_acc = prediction_acc(y_val_pred, labels)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(test_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(test_loader))


        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(test_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(test_loader):.3f}')

        model_path = 'model.pth'
        torch.save(model, model_path)

        
if __name__ == "__main__":
    
    parser.add_argument(
        "--train_file", type=str, default='training_data.pkl', help="train data file path"
    )
    
    args = parser.parse_args()
    print(args)
    
    with open(args.train_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    Y = data['Y']
    #assert type(X) == 'numpy.ndarray',  'Data load error'
    #assert type(Y) == 'numpy.ndarray',  'Data load error'
    
    #One hot encoding for Y
    Y = Y.flatten()
    Y = pd.get_dummies(Y)
    Y = Y.to_numpy()
    
    #Dictionary to store accuracy and loss stats
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }


    model = Net(num_feature = X.shape[1], num_class=NUM_CLASS)
    #print(model)
    
    #Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    #Create dataloader
    train_data = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
    test_data = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

    
    #Initialize model
    model = Net(num_feature = X.shape[1], num_class=NUM_CLASS)
    #print(model)
    
    #Train the model and save 
    train(model, train_loader, test_loader)
    

    




