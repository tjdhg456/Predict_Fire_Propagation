import os
import natsort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import util
from datetime import datetime

# path (/media/server/SNUdataset/fire_dataset/ERA5)
TE_CA_PATH = '/camdata2/ERA5/hourly/t2m/T_CA'
TE_AK_PATH = '/camdata2/ERA5/hourly/t2m/T_AK'
RH_CA_PATH = ''
RH_AK_PATH = ''
RA_CA_PATH = ''
RA_AK_PATH = ''
UW_CA_PATH = ''
UW_AK_PATH = ''
VW_CA_PATH = ''
VW_AK_PATH = ''


# image dimension
CA_X = 67
CA_Y = 73
AK_X = 113
AK_Y = 55

# Undefined value
UNDEFINED_VALUE = -9.99e+8


# Load DATA
def Load_Data(path, x_dim, y_dim):
    read_FILE = [file for file in os.listdir(path) if file.endswith(".bin")]
    read_FILE = natsort.natsorted(read_FILE)
    #tdim_list = list(map(lambda x: 366*24 if x%4 == 0 else 365*24, [int(f.split('_')[2].split('.')[0]) for f in read_FILE]))
    tdim_list = list(map(lambda x: 366 if x%4 == 0 else 365, [int(f.split('_')[2].split('.')[0]) for f in read_FILE]))

    array = []
    for file, t_dim in zip(read_FILE, tdim_list):
        with open(os.path.join(path,file),'rb') as f:
            data = np.fromfile(f, dtype=np.float32, count=x_dim*y_dim*t_dim)
            data = np.transpose(np.reshape(data,[t_dim,y_dim,x_dim]), [0,2,1])
        array.append(data)

    return np.concatenate(array, axis=0)


# Normalize
def MinMaxScaler(data, name=None, min=None, max=None):

    if min or max:
        numerator = data - min
        denominater = max - min
        return numerator / (denominater +1e-8)

    else:
        min, max = np.min(data), np.max(data)
        numerator = data - min
        denominater = max - min
        print('Range of {:4} is ({:.6}, {:.6})'.format(name, min, max))
        return (numerator / (denominater +1e-8), min, max)


# Null values
def Fill_UNDEF(x, undef, name):
    x[x==undef] = np.NaN
    count = np.sum(np.isnan(x))
    print('{:8} has {:8d} null Values'.format(name, count))

    if count > 0:
        x[np.isnan(x)] = 0.0

    return x


# Turn image (visualization)
def ruota_antiorario(matrix):
    ruota=list(zip(*reversed(matrix)))
    return[list(elemento) for elemento in ruota]
def ruota_orario(matrix):
    ruota=list(zip(*reversed(matrix)))
    return[list(elemento)[::-1] for elemento in ruota][::-1]



if __name__ == "__main__":

    # Load DATA
    t_ca = Load_Data(TE_CA_PATH, CA_X, CA_Y)
    t_ak = Load_Data(TE_AK_PATH, AK_X, AK_Y)
    #h_ca = Load_Data(RH_CA_PATH, CA_X, CA_Y)
    #h_ak = Load_Data(RH_AK_PATH, AK_X, AK_Y)
    #p_ca = Load_Data(RA_CA_PATH, CA_X, CA_Y)
    #p_ak = Load_Data(RA_AK_PATH, AK_X, AK_Y)
    #u_ca = Load_Data(UW_CA_PATH, CA_X, CA_Y)
    #u_ak = Load_Data(UW_AK_PATH, AK_X, AK_Y)
    #v_ca = Load_Data(VW_CA_PATH, CA_X, CA_Y)
    #v_ak = Load_Data(VW_AK_PATH, AK_X, AK_Y)

    #data_CA = [t_ca, h_ca, p_ca, u_ca, v_ca]
    #data_AK = [t_ak, t_ak, p_ak, u_ak, v_ak]
    data_CA = [t_ca]
    data_AK = [t_ak]

    # Date Index
    df_idx = pd.DataFrame({'Hours': pd.date_range('1979-01-01', '2019-01-01', freq='1H', closed='left')})

    
    # Check undefined values & normalize
    #for ca, ak, var in zip(data_CA, data_AK, ['temp', 'rhum', 'rain', 'uwnd', 'vwnd']):
    for ca, ak, var in zip(data_CA, data_AK, ['temp']):
        Fill_UNDEF(ca, UNDEFINED_VALUE, name=var)
        Fill_UNDEF(ak, UNDEFINED_VALUE, name=var)

        MinMaxScaler(ca, name=var)
        MinMaxScaler(ak, name=var)

         
    # Visualization
    plt.figure(figsize=(5,5))
    plt.imshow(ruota_orario(t_ca[0,:,:]), cmap=plt.cm.Spectral)
    plt.colorbar()
    plt.show()




