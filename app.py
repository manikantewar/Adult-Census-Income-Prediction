import pickle
from flask import Flask,redirect,url_for,render_template,app,request,jsonify
import pandas as pd
import numpy as np


app=Flask(__name__)

model=pickle.load(open('ADB_pipe.pkl','rb'))

@app.route('/')
def adult():
    return render_template('adult.html')

@app.route('/adult_api',methods=['POST'])
def adult_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=(np.array(list(data.values())).reshape(1,-1))
    print(new_data)
    output=model.predict(new_data)[0]
    print(output)
    return jsonify(int(output))


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    res=""
    if output==1:
        res=" Adult Income Salary is greater than 50k"
    else:
        res="Adult Income Salary is lesser than 50k"

    return render_template('predict.html',result="Prediction is {} so {} ".format(output,res))


if __name__=='__main__':
    app.run(debug=True)