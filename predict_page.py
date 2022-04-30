#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:01:22 2022

@author: dhruwin
"""
import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
        return data

data = load_model()

regressor_loaded = data['model']
hours = data['hours']


def show_predict_page():
    st.title('Students score Prediction')
    st.write('''### We need some information to predict the marks ###''')
    hrs = st.text_input("Enter number of hours you study :","")
    ok = st.button('Calculate Marks')
    if ok:
        X = np.array([[hrs]])
        X = X.astype(float)
        
        marks = regressor_loaded.predict(X)
        st.subheader(f'The estimated Marks are {marks[0]:.2f}')
        
show_predict_page()

        
    
