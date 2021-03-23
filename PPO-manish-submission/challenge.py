import os
import numpy as np
import random
import torch
from clf_dataset import ClassifierDataset

BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpecPredict:
    def __init__(self, model):
        self.model = torch.load(model)
        #self.mode = 'inference'
    
    def predict(self,test_data):
        '''Function to predict given test data'''
        return self.predict_batch(test_data)
    
        '''if(test_data.shape[1] == 1):
            return self.predict_sample(test_data)
        else:
            return self.predict_batch(test_data)'''

    
    def predict_sample(self, test_sample):
        '''Function to predict single sample'''
        #Some error in dimensions. Skipped using this
        self.model.eval()
        torch_sample = torch.from_numpy(test_sample).float().to(device)
        y_pred = self.model(torch_sample)
        y_pred = torch.log_softmax(y_pred, dim = 1)
        _, prediction = torch.max(y_pred, dim = 1)
        prediction = prediction.detach().cpu().numpy()[0]
        return prediction

    def predict_batch(self, test_batch):
        '''Function to predict for a batch'''
        Y_test = np.zeros((test_batch.shape[0],1)) #Dummy label to pass to dataloader
        test_data = ClassifierDataset(torch.from_numpy(test_batch).float(), torch.from_numpy(Y_test).long())
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

        with torch.no_grad():
            self.model.eval()
            pred = []
            for x_batch,_ in test_loader:
                x_test = x_batch.to(device)
                y_test_pred = self.model(x_test)
                y_test_pred = torch.log_softmax(y_test_pred, dim = 1)
                _, prediction = torch.max(y_test_pred, dim = 1)
                prediction = prediction.detach().cpu().numpy()
                pred.append(prediction)

            pred= [i for sublist in pred for i in sublist]


        return np.array(pred)