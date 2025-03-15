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
        
        # print(self.weights)
        # print(f'Shape: {x_data_wb.shape[0]}')
        l = 0
        epoch_error = []
        for epochs in range(self.max_epochs):
            tot_error = 0
            for batch in range(self.batch_size, x_data.shape[0], self.batch_size):

                batch_data = x_data_wb[l : batch]
                y_hat = self.predict(batch_data)
                error = self.MSE(y_hat, y_data[l : batch])
                #print(f'{l}: {error}')
                self.weights = self.weights - self.alpha * self.grad_descent(batch_data, y_hat, y_data[l : batch])
                #print(self.weights)
    
                l = batch
                tot_error += error
            #epoch_error.append(tot_error/(1338/50))
        #print(epoch_error)

    def predict(self, data_set):

        predicted_vals = data_set @ self.weights.T 
        return predicted_vals

    def MSE(self, predicted_vals, true_vals):
        
        mse_val = np.sum((predicted_vals - true_vals)**2)
        return mse_val

    def grad_descent(self, data_set, predicted_vals, true_vals):
        diff = predicted_vals - true_vals
        #print(diff)
        mse_der = np.sum(2*(diff) @ data_set)
        return mse_der
        