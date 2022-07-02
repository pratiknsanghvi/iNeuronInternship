# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:51:50 2022
@author: pratiksanghvi
"""
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template,redirect,url_for

app = Flask(__name__)
pickle_in = open("RandomForest.pkl", "rb")
rf_regressor = pickle.load(pickle_in)

pickle_in_knn = open("knn_regressor.pkl", "rb")
regressor = pickle.load(pickle_in_knn)

pickle_in_lasso = open("LassoRegression.pkl", "rb")
lasso_regressor = pickle.load(pickle_in_lasso)

pickle_in_lassoCV = open("LassoRegressionCV.pkl", "rb")
lassoCV_regressor = pickle.load(pickle_in_lassoCV)

pickle_in_linear = open("LinearRegression.pkl", "rb")
linear_regressor = pickle.load(pickle_in_linear)


@app.route('/', methods=["GET","POST"])
def selectMLAlgo():
    #selected = request.args.get([)'option']
    selectedVal = request.form.get("option",True)

    if request.method =='POST':
        if selectedValue == "rdf":
            return render_template("rdf.html")
        elif selectedValue == 'knn':
            return render_template("knn.html")
        elif selectedValue == 'lasso':
            return render_template("lasso.html")
        elif selectedValue == 'lassoCV':
            return render_template("lassoCV.html")
        elif selectedValue == 'linear':
            return render_template("linear.html")
        #return redirect (url_for('click',selectedValue=selectedVal))

    return render_template('MLType.html')


#@app.route('/<selectedValue>')
#def click(selectedValue):


def commonMethodForDataAssigning():
    features = []
    for x in request.form.values():
        features.append(x)

    age = features[0]
    bmi = features[1]
    sex = features[2]
    children = features[3]
    smoker = features[4]
    region = features[5]
    print(features)
    # ====convert to dummies data frame for the user input
    # Prepare the dataset for the model to predict
    df = pd.DataFrame({"age": age, "sex": 0, "bmi": bmi, "children": 0, "smoker": 0, "region": 0}, index=[0])

    if (sex.lower() == "male"):
        df['sex'] = df['sex'].replace([0], 1)

    if (smoker.lower() == "yes"):
        df['smoker'] = df['smoker'].replace([0], 1)

    if (children == '0'):
        df['children'] = df['children'].replace([0], 1)
    elif (children == '1'):
        df['children'] = df['children'].replace([0], 1)
    elif (children == '2'):
        df['children'] = df['children'].replace([0], 1)
    elif (children == '3'):
        df['children'] = df['children'].replace([0], 1)
    elif (children == '4'):
        df['children'] = df['children'].replace([0], 1)
    elif (children == '5'):
        df['children'] = df['children'].replace([0], 1)

    if (region.replace(" ", "").lower() == "southeast"):
        df['region'] = df['region'].replace([0], 1)
    elif (region.replace(" ", "").lower() == "northeast"):
        df['region'] = df['region'].replace([0], 2)
    elif (region.replace(" ", "").lower() == "northwest"):
        df['region'] = df['region'].replace([0], 3)

    return df


@app.route('/predict_lassoRegression', methods=["GET","POST"])
def predict_post_lassoRegression():
    df1 = commonMethodForDataAssigning()
    prediction = lasso_regressor.predict(commonMethodForDataAssigning())
    print(prediction)
    x = np.ascontiguousarray(commonMethodForDataAssigning(), dtype=int)
    prediction = lasso_regressor.predict(x)

    return render_template('lasso.html',
                           prediction_text='Employee Insurance predicted to be $ {}'.format(str(prediction)))

@app.route('/predict_lassoCVsRegression', methods=["GET","POST"])
def predict_post_lassoCVRegression():
    df1 = commonMethodForDataAssigning()
    prediction = lassoCV_regressor.predict(df1)
    print(prediction)
    x = np.ascontiguousarray(df1, dtype=int)
    prediction = lassoCV_regressor.predict(x)

    return render_template('lassoCV.html',
                           prediction_text='Employee Insurance predicted to be $ {}'.format(str(prediction)))

@app.route('/predict_linearRegression', methods=["GET","POST"])
def predict_post_linearRegression():
    df1 = commonMethodForDataAssigning()
    prediction = linear_regressor.predict(df1)
    print(prediction)
    x = np.ascontiguousarray(df1, dtype=int)
    prediction = linear_regressor.predict(x)

    return render_template('linear.html',
                           prediction_text='Employee Insurance predicted to be $ {}'.format(str(prediction)))

@app.route('/predict_RandomForest', methods=["GET","POST"])
def predict_post_randomForest():
    df1 = commonMethodForDataAssigning()
    # age = request.args.get("Age")
    # bmi = request.args.get("bmi")
    # sex = request.args.get("sex")
    # children = request.args.get("children")
    # smoker = request.args.get("smoker")
    # region = request.args.get("region")

    prediction = rf_regressor.predict(df1)
    print(prediction)
    x = np.ascontiguousarray(df1, dtype=int)
    prediction = rf_regressor.predict(x)

    return render_template('rdf.html',
                           prediction_text='Employee Insurance predicted to be $ {}'.format(prediction))


@app.route('/predict_knn_post', methods=["GET","POST"])
def predict_post_knn():

    features = []
    for x in request.form.values():
        features.append(x)

    age = features[0]
    bmi = features[1]
    sex = features[2]
    children = features[3]
    smoker = features[4]
    region = features[5]
    print(features)
    # ===================================================
    # ====convert to dummies data frame for the user input
    # Prepare the dataset for the model to predict
    df = pd.DataFrame({"age": age, "bmi": bmi, "sex_female": 0, "sex_male": 0, "children_0": 0,
                       "children_1": 0, "children_2": 0, "children_3": 0,
                       "children_4": 0, "children_5": 0,
                       "smoker_no": 0, "smoker_yes": 0,
                       "region_northeast": 0, "region_northwest": 0, "region_southeast": 0, "region_southwest": 0},
                      index=[0])

    if (sex.lower() == "male"):
        df['sex_male'] = df['sex_male'].replace([0], 1)
    elif (sex.lower() == "female"):
        df['sex_female'] = df['sex_female'].replace([0], 1)

    if (children == 0):
        df['children_0'] = df['children_0'].replace([0], 1)
    elif (children == 1):
        df['children_1'] = df['children_1'].replace([0], 1)
    elif (children == 2):
        df['children_2'] = df['children_2'].replace([0], 1)
    elif (children == 3):
        df['children_3'] = df['children_3'].replace([0], 1)
    elif (children == 4):
        df['children_4'] = df['children_4'].replace([0], 1)
    elif (children == 5):
        df['children_5'] = df['children_5'].replace([0], 1)

    if (smoker.lower() == "yes"):
        df['smoker_yes'] = df['smoker_yes'].replace([0], 1)
    elif (smoker.lower() == "no"):
        df['smoker_no'] = df['smoker_no'].replace([0], 1)

    if (region.replace(" ", "").lower() == "northeast"):
        df['region_northeast'] = df['region_northeast'].replace([0], 1)
    elif (region.replace(" ", "").lower() == "northwest"):
        df['region_northwest'] = df['region_northwest'].replace([0], 1)
    elif (region.replace(" ", "").lower() == "southeast"):
        df['region_southeast'] = df['region_southeast'].replace([0], 1)
    elif (region.replace(" ", "").lower() == "southwest"):
        df['region_southwest'] = df['region_southwest'].replace([0], 1)

    # ==========================================================
    print(df)

    prediction = regressor.predict(df)
    print(prediction)

    return render_template('knn.html',
                           prediction_text='Employee Insurance predicted to be $ {}'.format(prediction))


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8000, debug=True) - run on local machine
    app.run(debug=True)
