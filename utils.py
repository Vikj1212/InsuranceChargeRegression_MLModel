import numpy as np
import pandas as pd

class LinearRegression0:
    def __init__(self, batch_size=50, max_epochs=5, patience=3, alpha=0.1) -> None:
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.alpha = alpha
        pass

    def fit(self, x_data, y_data):
        bias_column = np.ones(x_data.shape[0])
        x_data_wb = x_data
        x_data_wb.insert(0, 'bias_val', bias_column)
        x_data_wb = x_data_wb.to_numpy()
        self.weights = np.random.rand(x_data_wb.shape[1])
        
        l = 0
        epoch_error = []
        for epochs in range(self.max_epochs):
            tot_error = 0
            for batch in range(self.batch_size, x_data.shape[0], self.batch_size):

                batch_data = x_data_wb[l : batch]
                y_hat = self.predict(batch_data)
                error = self.MSE(y_hat, y_data[l : batch])
                self.weights = self.weights - self.alpha * self.grad_descent(batch_data, y_hat, y_data[l : batch])
                #print(self.weights)
    
                l = batch
                tot_error += error
            print(f'Epoch {epochs} error: {error} ')
            #epoch_error.append(tot_error/(1338/50))
        #print(epoch_error)

    def predict(self, data_set):
        #print(f'data_set: {data_set.shape}\nWeights: {self.weights.reshape(-1, 1).shape}')
        predicted_vals = data_set @ self.weights.T
        #print(predicted_vals.reshape(-1, 1).shape) 
        return predicted_vals

    def MSE(self, predicted_vals, true_vals):
        mse_val = np.sum((predicted_vals.reshape(-1, 1) - np.array(true_vals).reshape(-1, 1))**2)
        if np.isnan(predicted_vals).any() or np.isnan(true_vals).any():
            print("Warning: NaN values detected in data!")
        if np.isinf(np.array(true_vals)).any() or np.isinf(predicted_vals).any():
            print("Warning: inf values detected in data!")
        #print(f'mse_val: {mse_val}')
        return mse_val

    def grad_descent(self, data_set, predicted_vals, true_vals):
        diff = np.array(predicted_vals - true_vals)
        #print(f'diff = {diff.reshape(-1, 1).shape}')
        #print(f'data_set.shape: {data_set.shape}')
        mse_der = (2*(data_set.T @ diff))/50
        return mse_der
        