import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
tf = TfidfVectorizer()

#load dump file
sentiment_model = joblib.load(open('sentiment_analysis_model.pkl','rb'))
tdump = joblib.load(open('tfidf_vectorizer.pkl','rb'))
encoder = LabelEncoder()
encoder.classes_ = np.load('label_classes.npy',allow_pickle=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_API',methods = ['POST'])
def predict_api():
    data = request.form['text'] 
    logging.warning("format of the data")
    pro_text = [data]
    output = sentiment_model.predict(tf.transform(pro_text).toarray())
    sentiment = encoder.inverse_transform(output)
    return jsonify({'sentiment': sentiment[0]})

@app.route('/predict_API',methods = ['POST'])
def predict():
    encoder = LabelEncoder()
    data = request.form['text'] 
    logging.warning(f"format of the data {data}")
    pro_text = [data]
    output = sentiment_model.predict(tf.transform(data).toarray())
    sentiment = encoder.inverse_transform(output)
    return render_template("home.html",prediction_text = "{} ".format(sentiment[0]))
    

if __name__ == '__main__':
    app.run(debug=True,port=5001)

