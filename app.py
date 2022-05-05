from flask import Flask
import pandas as pd
import numpy as np
import json
import os
from flask import request, session
from flask_cors import CORS
import joblib
import sklearn

app = Flask(__name__)
CORS(app, expose_headers='Authorization')
app.secret_key = os.urandom(24)

filename = 'logreg.model'
lreg = joblib.load(filename)

@app.route("/")
def index():
	print("Started")
	return "<h1>Welcome to Heart Attack Prediction server</h1>"

@app.route("/data", methods=["GET","POST"])
def data():
	list_data=[]
	#print(request.get_json());
	names = ["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"]
	for i in names:
		if i=="oldpeak":
			list_data.append(float(request.json[i]))
		else:
			list_data.append(int(request.json[i]))
	#print(list_data)
	pred = lreg.predict(np.array(list_data).reshape(1,13))
	print(pred)
	if int(pred)==1:
		return {"response":"More chance of heart attack"}
	else:
		return {"response":"Less chance of heart"}
if __name__ == "__main__":
	app.run()


