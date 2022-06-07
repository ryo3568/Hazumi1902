from sklearn.preprocessing import StandardScaler 

class Standardize:

    def __init__(self):
        self.scaler = StandardScaler() 

    def fit(self, x_train):
        self.scaler.fit(x_train.reshape(-1, 1463)) 

    def transform(self, data):
        batch_size = len(data) 
        data = self.scaler.transform(data.reshape(-1, 1463)) 
        return data.reshape(batch_size, -1, 1463)