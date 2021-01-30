# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:29:51 2021

@author: User
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import os
import platform
from IPython.display import Audio

def import_and_clean():
    df = pd.read_csv('training_data.csv', header=0)
    for col in df.columns:
        plt.hist(df.loc[df['label'] == 'female', col],label ="female")
        plt.hist(df.loc[df['label'] == 'male', col],label="male")
        plt.title(col)
        plt.xlabel("Feature magnitude")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.show()
    df = df[['meanfun' , 'minfun', 'maxfun','label']]
    df['label'] = df['label'].map({'female': 0, 'male': 1}).astype(int)
    return df

def parameter_tuning_svm(input_df):
    #x = input_df.iloc[:,:-1].values
    x = input_df[['meanfun' , 'minfun', 'maxfun']].values
    y = input_df['label'].values
    svc = SVC(kernel='linear')

    #segmenting data set and cross validation
    training, testing, training_result, testing_result = train_test_split(x, y, test_size=0.4, random_state=1)
    #回傳各fold的準確度
    scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')   
    print(scores)
    print(scores.mean())

    #Tuning C value ， c越大對svm的懲罰越大
    c_vals = list(range(1,30))
    #print(c_vals)
    accuracy_vals = []
    for val in c_vals:
        svc = SVC(kernel='linear', C=val)
        scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')
        accuracy_vals.append(scores.mean())

    plt.plot(c_vals, accuracy_vals)
    plt.xticks(np.arange(0,30,2))
    plt.xlabel('C values')
    plt.ylabel('Mean Accuracies')
    plt.show()

    optimal_cval = c_vals[accuracy_vals.index(max(accuracy_vals))]
    print(optimal_cval)

    #gamma value tuning
    gamma_vals = [0.00001,0.0001,0.001,0.01,0.1]
    accuracy_vals = []
    for g in gamma_vals:
        svc = SVC(kernel='linear', C=optimal_cval, gamma=g)
        scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')
        accuracy_vals.append(scores.mean())
    #
    plt.plot(gamma_vals, accuracy_vals)
    plt.xlabel('Gamma Values')
    plt.ylabel('Mean Accuracies')
    plt.show()

    optimal_gamma = gamma_vals[accuracy_vals.index(max(accuracy_vals))]
    print(optimal_gamma)

    svc = SVC(kernel='linear', C=optimal_cval, gamma=optimal_gamma)
    svc.fit(training, training_result)
    testing_predict = svc.predict(testing)
    print(metrics.accuracy_score(testing_predict, testing_result))

    svc = SVC(kernel='linear', C=optimal_cval, gamma=optimal_gamma)
    svc.fit(x,y)
    return svc

def record_audio():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "test_audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == '__main__':

    record_audio()
    if platform.system() == 'Linux':
        os.system('"Praat.exe" --run extract_freq_info.praat')
    elif platform.system() == 'Windows':
        print('window')
        #os.system(get)
        os.system('"Praat.exe" --run extract_freq_info.praat')
    else:
        os.system('"Praat.app/Contents/MacOS/Praat" --run extract_freq_info.praat')
    file = open('output.txt','r')
    values = file.readline()
    values = values.split(', ')
    for x in range(0,3):
        values[x] = float(values[x])/1000

    print("training and tuning svm")
    df =import_and_clean()
    tuned_svm = parameter_tuning_svm(df)
    predictions = tuned_svm.predict([values])
    print(predictions)
    if predictions == 0:
        print("you are a female")
    else:
        print("you are a male")