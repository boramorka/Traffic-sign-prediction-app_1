<h1 align="center">
  <br>
 Traffic sign prediction app deployed on Stramlit Cloud <br>
 <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/traffic-sign/streamlit-logo.svg" height="50">
</h1>


<h3 align="center">
  Built with
  <br>
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" height="30">
    <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/tensorflow.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/keras.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/matplotlib.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/numpy.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/pandas.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/scikit-learn.svg" height="30">
    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=green" height="30">
</h3>

<p align="center">
  <a href="#how-to-use">How To Use</a> •
  <a href="#how-to-run-locally">How To Run Locally</a> •
  <a href="#built-process">Built process</a> •
  <a href="#feedback">Feedback</a>
</p>

App link: https://boramorka-traffic-sign-prediction-app-1-appmain-app-pxaza3.streamlitapp.com/

## How To Use


## How To Run Locally

  ``` bash
  # Clone this repository
  $ git clone https://github.com/boramorka/Traffic-sign-prediction-app_1.git

  # Go into the repository
  $ cd Traffic-sign-prediction-app_1

  # Load traffic_sign_model.h5 from app folder and test it using Keras API
  ```

## Built process

Development process described in Traffic sign prediction model building jupyter notebook

Main notes:

- Libraries:
  ```python
  # Importing libraries
  import os
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib.image import imread
  import seaborn as sns
  import random
  from PIL import Image
  from sklearn.model_selection import  train_test_split
  from tensorflow.keras.utils import to_categorical
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
  ```
- Device: NVIDIA GeForce RTX 2060
- Model Architectue:
  ```python
  model = Sequential()

  model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(43, activation = 'softmax'))
  ```
- Total params: 445 803

## Conclusion
We started with downloading the dataset, preprocessing it, created the model and found out the predictions using the model. During preprocessing we found that this dataset has 43 classes. Model reached an accuracy of 95%+ in just 50 epochs, we can further optimize the model using hyper parameter tuning and reach a higher accuracy. 

## Scope
This model can be used in self driving cars which will enable them to automatically recognize traffic signs similarly the driver alert system inside cars will help and protect drivers by understanding the traffic signs around them.

## Feedback
:person_in_tuxedo: Feel free to send me feedback on [Telegram](https://t.me/boramorka). Feature requests are always welcome. 

:abacus: [Check my other projects.](https://github.com/boramorka)


