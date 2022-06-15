import glob
import re 
import numpy as np  
import pandas as pd

def get_id(file):
    pattern = '../data/dumpfiles/([^.]*).csv'
    id = re.search(pattern, file)
    return id

def get_Labels(df):
    return df['TS_ternary'].values.tolist()

def get_Text(df):
    res = []
    for index, row in df.iterrows():
        res.append(df.loc[index, 'word#001':'word#967'].values)
    return res

def get_Audio(df):
    res = []
    for index, row in df.iterrows():
        res.append(df.loc[index, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values)
    return res 

def get_Visual(df):
    res = []
    for index, row in df.iterrows():
        res.append(df.loc[index, '17_acceleration_max':'AU45_c_mean'].values)
    return res

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

class Hazumi:

    def __init__(self, test_file):
        FILE_PATH = '../data/dumpfiles/*.csv'
        self.files = glob.glob(FILE_PATH)
        self.test_file = test_file

    def load(self):
        videoLabels = {} 
        videoText = {} 
        videoAudio = {} 
        videoVisual = {} 
        trainVid = set()
        testVid = set()

        for file in self.files:
            df = pd.read_csv(file)
            id = get_id(file)
            videoLabels[id] = get_Labels(df)
            videoText[id] = get_Text(df)
            videoAudio[id] = get_Audio(df) 
            videoVisual[id] = get_Visual(df) 
            
            if file == self.test_file:
                testVid.add(id)
            else:
                trainVid.add(id)

        return videoLabels, videoText, videoAudio, videoVisual, trainVid, testVid

