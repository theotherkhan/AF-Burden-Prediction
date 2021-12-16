import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import wfdb
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, MaxPooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import time
from keras import metrics

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "/Users/Hasan/Desktop/Workspace/AF-Burden-Prediction/Project/data"

def load_record(sample_path):
    
    '''  returns signal, global label, local labels ''' 
    
    sig, fields = wfdb.rdsamp(sample_path)
    ann_ref = wfdb.rdann(sample_path, 'atr')
    
    label = fields['comments'][0]
    fs = fields['fs']
    sig = sig[:, 1]
    length = len(sig)
    
    beat_loc = np.array(ann_ref.sample) # r-peak locations
    ann_note = np.array(ann_ref.aux_note) # rhythm change flag
    
    return sig, length, fs, label, ann_note, beat_loc

def normalize(signal):

    values = signal
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    normalized = [item for sublist in normalized for item in sublist]

    return normalized

def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def build_chunked_input(seconds):
        
    ''' Builds chunked signal input  '''
        
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()

    input_df = pd.DataFrame(columns=["Sequence Number", "Chunked Signal", "Sequence Label", 
                                     "Chunked Label", "Signal Length", "AF Burden"])
      
    for i, sample in enumerate(test_set):
        
        print(sample)
        #print(i, end='\r')
        sample_path = os.path.join(DATA_PATH, sample)
        sig, sig_len, fs, label, label_arr, beat_loc = load_record(sample_path)
        
        r_peaks = beat_loc
        #qrs_peaks = (qrs_detect(sig, fs)).astype(int)
        #r_peaks = qrs_peaks
        chunksize = seconds*200
        
        #print(sig)
        #print("\n\nr_peaks: ", r_peaks, len(r_peaks))
        #print("\nqrs_detect: ", qrs_peaks, len(qrs_peaks))
        
        sig = normalize(sig)
        
        ## Calculate exact AF ranges in sequence Label arr acts as an index 
        ## for which peaks in r_peaks are afib
           
        af_ranges = []
        af_range = []
                
        for li, l in enumerate(label_arr):
            if l == "(AFIB" or l == "(AFL":
                printlabelarr = True
                start = r_peaks[li] #r_peaks[min(len(r_peaks)-1, li)]
                af_range.append(start)
            if l == "(N":
                stop = r_peaks[li] #r_peaks[min(len(r_peaks)-1, li)]
                af_range.append(stop)
                af_ranges.append(af_range)
                af_range = []
        
        #print("\n", af_ranges)
        
        ## Label sections of signal as AF/non AF
        loc_labels = [0]*sig_len
        for rng in af_ranges:
            start = rng[0]
            stop = rng[1]
            loc_labels[ start : stop ] = [1] * (stop-start) 
        
        ## Break signal and label sequences down to n-second chunks
        chunked_sig = chunks(sig, chunksize)[:-1]
        chunked_label = chunks(loc_labels, chunksize)[:-1]
        
        ## Calculate AF Burden per sequence based on AF ranges
        burden=0
        for rng in af_ranges:
            burden+=rng[1]-rng[0] 
        burden = burden/sig_len  

        input_df.at[i, 'Sequence Number'] = i
        input_df.at[i, 'Chunked Signal'] = chunked_sig
        input_df.at[i, 'Chunked Label'] = chunked_label
        input_df.at[i, 'Signal Length'] = sig_len
        input_df.at[i, 'Sequence Label'] = label
        input_df.at[i, 'AF Burden'] = burden
        
    ## Convert from sequence-level df into chunk-level df
    input_df = input_df.explode(["Chunked Signal", "Chunked Label"])
    input_df.dropna(subset=['Chunked Label'], inplace=True)

    ## Assign most common granular label as overall chunk label
    input_df["Chunked Label"] = input_df["Chunked Label"].apply(lambda x: Counter(x).most_common(1)[0][0])
    
    return input_df

if __name__ == '__main__':

    seconds=30
    print("\nBuilding input df...")
    chunk_df = build_chunked_input(seconds)
    chunk_df.dropna(inplace=True)

    ## Split into train and test data (70-30 split). Ensure record chunks are not split between train and test

    print("\nBuilding train/test splits...")
    cutoff = int(.7*max(chunk_df['Sequence Number']))
    train_df = chunk_df[chunk_df['Sequence Number'] < cutoff]
    test_df = chunk_df[chunk_df['Sequence Number'] >= cutoff]

    X_train = pd.DataFrame(train_df['Chunked Signal'].tolist())
    X_test = pd.DataFrame(test_df['Chunked Signal'].tolist())

    y_train = pd.DataFrame(train_df['Chunked Label'].tolist())
    y_test = pd.DataFrame(test_df['Chunked Label'].tolist())

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    ## Build network
    
    batch = 16
    epochs = 3
    shape = np.size(X_train, 1)
    filters = (100, 160)
    drop = 0.3

    model = Sequential()
    model.add(Dense(filters[0], activation='relu', input_shape = (shape,1)))
    model.add(Conv1D(filters[0], 10, activation='relu'))
    model.add(Conv1D(filters[0], 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(filters[1], 10, activation='relu'))
    model.add(Conv1D(filters[1], 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid')) 
    model.summary()
    model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

    X_train = np.expand_dims(X_train, 2)

    print("Number of layers: ", len(model.layers))

    print("\nTraining CNN...")
    model.fit(X_train,y_train, batch_size = batch, epochs = epochs)
    try:
        X_test_expand = np.expand_dims(X_test, 2)
        score = model.evaluate(X_test_expand, y_test, batch_size = batch)
    except:
        score = model.evaluate(X_test, y_test, batch_size = batch)

    ## Saving Model
    
    print("\nSaving model...")
    model_name = "model_filt" + str(filters[0]) + "_layers" + str(len(model.layers)) + "_sec" + str(seconds) + ".json"
    model_weights = "weights_filt" + str(filters[0]) + "_layers" + str(len(model.layers)) + "_sec" + str(seconds) + ".h5"
    model_json = model.to_json()
    with open(model_name, "w") as json_file:
        json_file.write("model_json")
    model.save_weights(model_weights)
    print("Saved model and weights to disk")

    print("\nPredicting on test data...")

    threshold = 0.5
    y_pred = model.predict(X_test, batch_size = batch)
    y_test['y_pred'] = y_pred[:,0] > threshold
    if "index" in y_test.columns: y_test.drop('index', inplace=True)
    y_test.columns=['y_true', 'y_pred']
    y_test['y_pred'] = y_test['y_pred'].astype(float)

    cm = confusion_matrix(y_test['y_true'], y_test['y_pred'])
    
    precision, recall, _, _ = precision_recall_fscore_support(y_test['y_true'], y_test['y_pred'], average='macro')

    print("\n Accuracy: ", score[1])
    print("\n Loss: ", score[0])
    print("\n Precision: ", precision, " Recall: ", recall)
    print("\n F1 score: ", f1_score(y_test['y_true'], y_test['y_pred'], average='macro'))
    print("\n Confusion Matrix: \n", cm)

    ## Append predictions to test dataset

    test_df.reset_index(inplace=True)
    y_test.reset_index(inplace=True)
    chunked_test = pd.concat([test_df, y_test], axis=1)

    ## Condense dataset back to the sequence level, calculate Predicted AF Burden

    chunked_test['y_pred_2'] = chunked_test['y_pred']

    seq_test = (chunked_test.drop(columns=['Chunked Signal'])
          .groupby(['Sequence Number', 'Sequence Label', 'AF Burden', 'Signal Length'])
          .agg({'Chunked Label': lambda x: x.tolist() , 'y_pred': lambda x: x.tolist(), 'y_pred_2':'sum', })
          .rename({'y_pred_2' : 'AF Episodes'},axis=1)
          .reset_index())

    seq_test['Predicted AF Burden'] = (seq_test['AF Episodes']*seconds*200) / seq_test['Signal Length']


    ## Calculate MAE for AF Burden Predications

    burden_mae = mean_absolute_error(seq_test['AF Burden'],seq_test['Predicted AF Burden'])
    print("Burden MAE: ", burden_mae)
    



