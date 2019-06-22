#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:16:02 2019

@author: spavko
"""

import requests
import json

url = 'http://0.0.0.0:5000/predict_pulsar/'

if __name__ == '__main__':
    
    ## ------------------------------
    data = [[140.5625,55.68378214,-0.234571412,-0.699648398,3.199832776,19.11042633,7.975531794,74.24222492]]
    j_data = json.dumps(data)
    
    print(j_data)
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    print(r, r.text)
    
    
    ## ------------------------------
    data2 = [[99.3671875,41.57220208,1.547196967,4.154106043,27.55518395,61.71901588,2.20880796,3.662680136]]
    j_data2 = json.dumps(data2)
    
    print(j_data2)
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data2, headers=headers)
    print(r, r.text)
    
    
    ## ------------------------------
    data3 = [[120.5546875,45.54990543,0.282923998,0.419908714,1.358695652,13.07903424,13.31214143,212.5970294]]
    j_data3 = json.dumps(data3)
    
    print(j_data3)
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data3, headers=headers)
    print(r, r.text)

