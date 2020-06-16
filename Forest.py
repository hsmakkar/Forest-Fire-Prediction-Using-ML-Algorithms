import numpy as np
import pandas as pd

data=pd.read_csv('forest_fires.csv')
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12)
                   ,inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7),inplace=True)
from sklearn import preprocessing
m=data.values
min_max_scaler=preprocessing.MinMaxScaler()
m_scaled=min_max_scaler.fit_transform(m)
data=pd.DataFrame(m_scaled)
data_array=np.array(data)

x=data_array[:,0:12]
y=data_array[:,12]

y=np.reshape(y,(517,1))

from sklearn.model_selection import train_test_split
#Splitting the Data into Training and Tesing Sets
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.3,random_state=42)


from sklearn import linear_model
from sklearn.metrics import mean_squared_error,accuracy_score
lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)

pred_y=lr.predict(x_test)

print("Mean Squared Error: ",mean_squared_error(y_test,pred_y))

#Making Another Model Known as Support Vector Machine


binary_area_values = []
count = 0

for value in y:
    if(value == 0):
        binary_area_values.append(0)
    else:
        binary_area_values.append(1)

train_x, test_x, train_y, test_y = train_test_split(x, binary_area_values, test_size=0.15, random_state = 4)

svm_model = svm.SVC(kernel='linear', gamma=100)
svm_model.fit(train_x, train_y)
predicted_y = svm_model.predict(test_x)

print("The predicted values are:", predicted_y)
print("The accuracy score is " + str(accuracy_score(test_y, predicted_y) * 100) + ".")
