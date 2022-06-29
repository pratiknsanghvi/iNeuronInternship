# -*- coding: utf-8 -*-
#pip install fastapi uvicorn

# Library Imports
import uvicorn  # ASGI
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from InsuranceParameters import InsPara

# 2. Create an Object
app = FastAPI()

pickle_in = open("knn_regressor.pkl", "rb")
regressor = pickle.load(pickle_in)


# Base page
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'This is a FASTAPI implementation'}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
# @app.get('/test')
# def get_name(name: str):
#   return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence


@app.post('/predictInsuranceRates')
def predict_insurancePrice(data: InsPara):
    data = data.dict()
    age = data['age']
    bmi = data['bmi']
    sex = data['sex']
    children = data['children']
    smoker = data['smoker']
    region = data['region']
    # ===================================================
    # ====convert to dummies data frame for the user input
    # Prepare the dataset for the model to predict
    df = pd.DataFrame({"age": age, "bmi": bmi, "sex_female": 0, "sex_male": 0, "children_0": 0,
                       "children_1": 0, "children_2": 0, "children_3": 0,
                       "children_4": 0, "children_5": 0,
                       "smoker_no": 0, "smoker_yes": 0,
                       "region_northeast": 0, "region_northwest": 0, "region_southeast": 0, "region_southwest": 0},
                      index=[0])

    if sex.lower() == "male":
        df['sex_male'] = df['sex_male'].replace([0], 1)
    elif sex.lower() == "female":
        df['sex_female'] = df['sex_female'].replace([0], 1)

    if children == '0':
        df['children_0'] = df['children_0'].replace([0], 1)
    elif children == '1':
        df['children_1'] = df['children_1'].replace([0], 1)
    elif children == '2':
        df['children_2'] = df['children_2'].replace([0], 1)
    elif children == '3':
        df['children_3'] = df['children_3'].replace([0], 1)
    elif children == '4':
        df['children_4'] = df['children_4'].replace([0], 1)
    elif children == '5':
        df['children_5'] = df['children_5'].replace([0], 1)

    if smoker.lower () == "yes":
        df['smoker_yes'] = df['smoker_yes'].replace([0], 1)
    elif smoker.lower() == "no":
        df['smoker_no'] = df['smoker_no'].replace([0], 1)

    if region.replace(" ","").lower()=="northeast":
        df['region_northeast']=df['region_northeast'].replace([0],1)
    elif region.replace(" ","").lower()=="northwest":
        df['region_northwest']=df['region_northwest'].replace([0],1)
    elif region.replace(" ","").lower()=="southeast":
        df['region_southeast']=df['region_southeast'].replace([0],1)
    elif region.replace(" ","").lower()=="southwest":
        df['region_southwest']=df['region_southwest'].replace([0],1)

    # ==========================================================
    print(df)
    x  = np.ascontiguousarray(df,dtype=int)
    # if global_optionValue == 'knn':
    prediction = regressor.predict(x)
    #prediction = regressor.predict(df)
    print(str(prediction))
    m = str(prediction)
    results = {'prediction': m}
    return results


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
# uvicorn knn_fastapi:app --reload
