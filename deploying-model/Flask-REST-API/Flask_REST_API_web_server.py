#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:11:09 2019

@author: spavko
"""

import numpy as np
import os
import pickle
import json
from PulsarClass import PulsarModel

from flask import Flask, request, jsonify
# from flask_restful import reqparse, abort, Api, Resource
      
# initialize the flask and api objects
app = Flask(__name__)
  

@app.route('/predict_pulsar/', methods=['POST'])
def return_pros():
    j_data = request.get_json()
    
    j_response = {}
    
    # needed .item() for the proper type conversion
    
    j_response["Pulsar probability"] = model.predict_proba(j_data)[0][1].item()
    j_response["Prediction class"] = model.predict(j_data)[0].item()
    
    j = json.dumps(j_response)
    
    return j

if __name__ == '__main__':
    # initialize the pulsar object
    model = PulsarModel()
    
    # load the model into the pulsar object   
    path = os.getcwd() + '/serialized_classifier.pkl'
    with open(path, 'rb') as f:
        model.model = pickle.load(f)

    # run the flask app        
    app.run(debug=True, host='0.0.0.0')  