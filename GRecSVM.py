# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:59:44 2021

@author: User
"""
from flask import Flask,render_template,request,url_for,redirect,make_response
import pyaudio
import wave
import os
import platform
from IPython.display import Audio
import pickle

app = Flask(__name__)

@app.route("/",methods=["GET","POST"]) #函式的裝飾，已函示為基礎，提供附加的功能
def home():

    if request.method == "POST":
        print("data receive post")
        af = request.files['file'].read()
        p = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        wf = wave.open("test_audio.wav", 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(44100)
        wf.writeframes(af)
        wf.close()
      
        if platform.system() == 'Linux':
            os.system('"Praat.exe" --run extract_freq_info.praat')
        elif platform.system() == 'Windows':
            os.system('"Praat.exe" --run extract_freq_info.praat')
        else:
            os.system('"Praat.app/Contents/MacOS/Praat" --run extract_freq_info.praat')
        file = open('output.txt','r')
        values = file.readline()
        values = values.split(', ')
        for x in range(0,3):
            values[x] = float(values[x])/1000
    
        with open('model/svm_model.pickle', 'rb') as f:
            tuned_svm = pickle.load(f)
            predictions = tuned_svm.predict([values])
            if predictions == 0:
                print("you are a female")
                result="female"
                #return render_template('home.html',result=result)
                #return render_template('predict.html',result=result)
                #return redirect(url_for('predict'))
                return make_response(result)
            else:
                print("you are a male")  
                result = "male"
                #return render_template('home.html',result=result)
                return make_response(result)
                #return render_template('home.html',result=result)
                #return redirect(url_for('predict'))
    else:
        result=None
        return render_template('home.html',result=result)
   

if __name__ =="__main__":
    #app.debug=True
    app.run()