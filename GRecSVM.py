# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:59:44 2021

@author: User
"""
from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route("/") #函式的裝飾，已函示為基礎，提供附加的功能
def home():
    return render_template('home.html')

@app.route("/test")
def test():
    return "this is test"

if __name__ =="__main__":
    app.run()