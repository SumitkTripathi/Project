from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("Model.pkl","rb"))
app = Flask(__name__)
df = pd.read_csv("Cleaned_Car_data.csv")

@app.route("/") 
def index():
    companies = sorted(df["company"].unique())
    companies.insert(0,'Select Company')
    car_models = sorted(df["name"].unique())
    fuel_type = df["fuel_type"].unique()
    year = sorted(df["year"].unique(),reverse=True)
    return render_template("index.html",companies_list=companies,car_models_list=car_models,years=year,fuel_types=fuel_type) 

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kilometer = int(request.form.get('kilometer'))

    prediction = model.predict(pd.DataFrame([[car_model,company,year,kilometer,fuel_type]],columns=["name","company","year","kms_driven","fuel_type"]))
    return str(np.round(prediction[0],2))
if __name__== "__main__":
    app.run(debug=True)
