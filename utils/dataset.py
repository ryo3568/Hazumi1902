import glob
import numpy as np  
import pandas as pd

class Hazumi1902:

    def __init__(self, target):
        FILE_PATH = '../data/dumpfiles/*.csv'
        files = glob.glob(FILE_PATH)
        self.files = files
        self.target = target

    def feature_extraction(self, df):
        col = ['start(exchange)[ms]', 'end(system)[ms]', 'end(exchange)[ms]',\
        'kinectstart(exchange)[ms]', 'kinectend(system)[ms]',\
        'kinectend(exchange)[ms]', 'SS_ternary', 'TC_ternary', 'TS_ternary', 'SS',\
        'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'TS1', 'TS2', 'TS3', 'TS4', 'TS5']
        
        df = df.drop(col, axis=1)
        return df.values.tolist()

    def target_extraction(self, df):
        if self.target == 1:
            df = (df.loc[:, 'TS1':'TS5'].sum(axis=1) > 20).astype('int')
        
        return df.values.tolist()

    def load_data(self, testfile): #list? ndarray?
        x_train = []
        x_test = []
        t_train = []
        t_test = []

        for file in self.files:
            df = pd.read_csv(file)
            df_x = self.feature_extraction(df) 
            df_t = self.target_extraction(df)

            if file == testfile:
                x_test.append(df_x)
                t_test.append(df_t)
            else:
                x_train.append(df_x) 
                t_train.append(df_t)
        
            
        return x_train, x_test, t_train, t_test