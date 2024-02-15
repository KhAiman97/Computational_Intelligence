from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import tkinter as tk

from GUI_Heart import HeartRatePredictionApp
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartRatePredictionApp(root)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate the x and y coordinates for the top-right corner
    x_coordinate = screen_width - 500  # Adjust 500 according to your window width
    y_coordinate = 120
    
    # Set the geometry of the window
    root.geometry(f"500x600+{x_coordinate}+{y_coordinate}")
    root.mainloop()
    self_info = app.self_info
    
test_df = pd.DataFrame([self_info], columns=['BMI', 'Cholesterol', 'Heart Rate', 'Stress Level', 'Triglycerides', 'Previous Heart Problems'])
data = pd.read_csv('Heart_train.csv')
test = pd.read_csv('Heart_test.csv')
Data = data.loc[:,['BMI','Cholesterol','Heart Rate','Stress Level','Triglycerides','Previous Heart Problems','Heart Attack Risk']]
test = test.loc[:,['BMI','Cholesterol','Heart Rate','Stress Level','Triglycerides','Previous Heart Problems','Heart Attack Risk']]

# digitizing continuous variable
aa = Data['Heart Attack Risk']  
minima = aa.min()
maxima = aa.max()
bins = np.linspace(minima-1,maxima+1, 3)
binned = np.digitize(aa, bins)
data_train, data_test = train_test_split(Data, test_size=0.2,
                                          random_state=101,stratify=binned)
    
X_train = data_train.drop("Heart Attack Risk",axis=1).values
y_train = data_train["Heart Attack Risk"].copy().values
X_test = data_test.drop("Heart Attack Risk",axis=1).values
y_test = data_test["Heart Attack Risk"].copy().values
X_val = test.drop("Heart Attack Risk",axis=1).values
y_val = test["Heart Attack Risk"].copy().values

scaler_x = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler_y = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler_x.fit(X_train)

test_scaled = scaler_x.transform(test_df)
print(test_scaled)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)
X_val = scaler_x.transform(X_val)

scaler_y.fit(y_train.reshape(-1,1))
y_train = scaler_y.transform(y_train.reshape(-1,1))
y_test = scaler_y.transform(y_test.reshape(-1,1))

from ANFIS import EVOLUTIONARY_ANFIS

E_Anfis = EVOLUTIONARY_ANFIS(functions=3,generations=50,offsprings=10,
                            mutationRate=0.2,learningRate=0.2,chance=0.7)

bestParam, bestModel=  E_Anfis.fit(X_train,y_train,optimize_test_data=False)

bestParam, bestModel = E_Anfis.fit(X_train,y_train,X_test,y_test,optimize_test_data=True)

testing = E_Anfis.predict(X_val ,bestParam, bestModel)
#print (testing) 
Maxval = np.max(testing)
Minval = np.min(testing)
Outputs= E_Anfis.predict(test_scaled ,bestParam, bestModel) 
#print(Outputs)
Outputs = (Outputs - Minval)/(Maxval - Minval)
print("Output : ", Outputs)

if Outputs >= 0.5:
    newOutput = 1
else:
    newOutput = 0

from Out_GUI import HeartRatePredictor
if __name__ == "__main__":
    #root = tk.Tk()
    predictor = HeartRatePredictor()
    predictor.set_prediction(newOutput)
    predictor.main()
