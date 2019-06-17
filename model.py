import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle 

data = pd.read_csv('salary_data.csv')

X = data.iloc[:,:2]

y=data.iloc[:,-1]

mod = LinearRegression()
mod.fit(X,y)

pickle.dump(mod, open('model.pkl','wb'))

model =pickle.load(open('model.pkl','rb'))
print(model.predict([[1,35]]))
