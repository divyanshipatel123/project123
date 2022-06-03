import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from PIL import Image
import PIL.ImageOps
import os 
import time, ssl

X = np.load('image.npz')["arr_0"]
Y = pd.read_csv("labels.csv")["labels"]
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n_classes = len(classes)
print(pd.Series(Y).value_counts())
print(len(X))
print(len(X[0]))
print(Y[0])
print(X[0])

X_train ,X_test , Y_train , Y_test = train_test_split(X , Y , test_size=2500 , train_size = 7500 , random_state = 0)
x_train_scale = X_train/255
x_test_scale = X_test/255

model = LogisticRegression(solver = "saga" , multi_class = "multinomial")
model.fit(x_train_scale , Y_train)
Y_pred = model.predict(x_test_scale)

accuracy = accuracy_score(Y_test , Y_pred)
print("Accuracy of the model is :" , accuracy)

cap = cv2.VideoCapture(0)
while (True):
    try :
        ret , frame = cap.read()
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        height , width = gray.shape()
        upperLeft = (int(width/2 - 56) , int(height/2 - 56))
        bottomRight = (int(width/2 + 56) , int(height/2 + 56))
        cv2.rectangle(gray , upperLeft , bottomRight , (0 , 255 , 0) , 2)
        roi = gray[upperLeft[1]: bottomRight[1]  , upperLeft[0]:bottomRight[0]]
        
        ImgPil = Image.fromarray(roi)
        ImgBW = ImgPil.convert("L")
        ImgBW_resized = ImgBW.resize((28 , 28) , Image.ANTIALIAS())
        ImgBW_resized_inverted = PIL.ImageOps.invert(ImgBW_resized)
        pixelFilter = 20
        Scaler_Image = np.percentile(ImgBW_resized_inverted , pixelFilter)
        ImgBW_resized_inverted_scaled = np.clip(ImgBW_resized_inverted - Scaler_Image , 0 , 255)
        max_pixel = np.max(ImgBW_resized_inverted)
        ImgBW_resized_inverted_scaled = np.asarray(ImgBW_resized_inverted_scaled/max_pixel)

        # the prediction
        test_sample = np.array(ImgBW_resized_inverted_scaled).reashape(1,784)
        test_pred = model.predict(test_sample)
        print("THE PREDICTED CLASS IS : " , test_pred)
        
        cv2.imshow("Frame", gray)
        if cv2.waitkey(1) & 0xFF == ord("q"):
            break
    except Exception as E :
        pass

cap.release()
cv2.destroyAllWindows()