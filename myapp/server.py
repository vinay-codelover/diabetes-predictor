# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:55:19 2020

@author: vijay
"""
from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
server = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@server.route('/')
def home():
    return render_template('index.html')
@server.route('/predict',methods = ['POST'])
def predict():
      int_features = [int(float(x)) for x in request.form.values()]
      final_features = [np.array(int_features)]
      float_features = [float(x) for x in request.form.values()]
      perc_features = [np.array(float_features)]
      prediction = model.predict(final_features)
      percentage = model.predict_proba(perc_features)
      output = prediction[0]
      return render_template('index.html', prediction_text="The probability of having diabetes is "+str(percentage[:,0][0])+"\n"+"it seems to be "+str(output)+'itive')
if __name__ == "__main__":
    server.run(debug=True)