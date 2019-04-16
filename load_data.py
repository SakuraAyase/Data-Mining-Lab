import pandas as pd
import numpy  as np



def load_data(path = 'data\segment.dat'):

    data = pd.read_table(path,header=None,sep='\s+')
    data =  data.values
    return data

def segment_label_data(dataset):
    dataset = np.array(dataset)
    label = dataset[:, -1:]
    data = dataset[:, :-1]
    return data,label

def load():
    return segment_label_data(load_data())
