import json
#ML imports
import datareader as datareader
import numpy as np
import pandas as pd
from pandas import read_csv
from numpy import set_printoptions
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
import joblib
import requests
import csv

import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Quick workarround for the datareader not showing
#import pip
#pip.main(['install', 'pandas_datareader'])

import pandas_datareader as pdr
import tkinter as tk
from tkinter import messagebox


#GUI Initialized
window = tk.Tk()
window.title("AI Investment Advisor")
frame = tk.Frame(master=window, width=800, height=800)
frame.pack()

header = tk.Label(master=frame, text = "Welcome to your Inversment advisor, AI will help you make the right investment descision", bg="white", fg="black", font=18)
header.pack()

frm_ticker = tk.Frame(bg='white', borderwidth=5)
frm_ticker.pack(fill=tk.BOTH)
lbl_ticker = tk.Label(master=frm_ticker, text= "Enter your desired Company ticker you want to see")
ent_ticker = tk.Entry(master=frm_ticker, width=5)
lbl_ticker.pack()
ent_ticker.pack()
def callback():
    tickerName = str(ent_ticker.get())

    if tickerName == 'APPL':
        return(tickerName)
    elif tickerName =='FB':
        return(tickerName)
    elif tickerName == 'GOOG':
        return(tickerName)

def prediction(tickName):
    predictionResult = MLPred(tickName)
    return(predictionResult)

def click_handler(event):
    tick = callback()
    predResult = prediction(tick)
    messagebox.showinfo("Prediction", "The predicted stock price for tomorow is: ", predResult)

frm_predict = tk.Frame(bg='green', borderwidth=2)
frm_predict.pack(fill=tk.BOTH, side =tk.BOTTOM)
btn_predict = tk.Button(master=frm_predict, text="Predict", fg="black", bg="green", width=10, height=3)
btn_predict.pack()
btn_predict.bind("<Button-1>", click_handler)


def MLPred(company_ticker):


    #Data loading
    comp = 'FB'
    start = datetime.datetime(2012,1,1)
    end = datetime.datetime(2020,1,1)

    data = pdr.DataReader(comp, 'yahoo', start, end)

    #print(data)
    print(data.shape)
    print(type(data))


    #Data preparation


    #scaler implementation
    sclr = MinMaxScaler(feature_range=(0,1))
    scaled_data = sclr.fit_transform(data['Close'].values.reshape(-1,1))

    pred_days = 60

    x_train = []
    y_train = []

    for x in range(pred_days, len(scaled_data)):
        x_train.append(scaled_data[x-pred_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    #Model


    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=32, batch_size=32)

    final_model = 'final_model.sav'
    #joblib.dump(model, final_model)



    f_model = 'final_model.sav'
    #teesting the program for past data
    #loaded_model = joblib.load(f_model)

    test_init = datetime.datetime(2020,1,1)
    test_end = datetime.datetime.now()

    test_data = pdr.DataReader(comp, 'yahoo', test_init, test_end)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - pred_days:].values

    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = sclr.transform(model_inputs)


    x_test = []

    for x in range(pred_days, len(model_inputs)):
        x_test.append(model_inputs[x-pred_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = sclr.inverse_transform(predicted_prices)

    '''
    #plot the test
    pyplot.plot(actual_prices, color="black", label=f"Actual")
    pyplot.plot(predicted_prices, color="blue", label=f"Predicted Price")
    pyplot.title(f"{comp} Share Price")

    pyplot.legend()
    pyplot.show()
    #print(predicted_prices)

    '''

    #Future predictions

    #loaded_model = joblib.load(f_model)
    actual_data = [model_inputs[len(model_inputs) - pred_days:len(model_inputs+1), 0]]

    actual_data = np.array(actual_data)
    actual_data = np.reshape(actual_data, (actual_data.shape[0], actual_data.shape[1],1))

    prediction = model.predict(actual_data)
    prediction = sclr.inverse_transform(prediction)

    print(f"Prediction for tomorw's closing price: {prediction}")

    predInt = prediction.astype(float)

    return(predInt)

window.mainloop()
