import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv('model_data.csv')
data = data.drop('Unnamed: 0',axis = 1)
X = data.drop('MRP',axis = 1) # MRP IS OUR TARGET
y = data['MRP'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

RFR = RandomForestRegressor()

RFR = RFR.fit(X_train,y_train)
preds = RFR.predict(X_test)

score = r2_score(preds,y_test)
print("Accuracy: {:.2f}%".format(score*100))
print(RFR.feature_names_in_)

joblib.dump(RFR, 'rfr_model.pkl')
