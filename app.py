from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
app = Flask(__name__)
model = pickle.load(open('xgb_regressor1.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index1.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            op= float(request.form['open'])
            volume=float(request.form['volume'])
            high=float(request.form['high'])
            low=float(request.form['low'])
            text=request.form['txt']
            sid = SentimentIntensityAnalyzer()
            compound=float(sid.polarity_scores(text)['compound'])
            X=np.array([[op,high,low,volume,compound]])
            print(X);
            prediction=model.predict(X);
            output=prediction[0];

        
            if volume<0 or op<0 or high<0 or low<0:
                return render_template('index1.html',prediction_text="Please,enter valid values")
            if  output<0:
                return render_template('index1.html',prediction_text="Sorry, we can't predict this.")
            else:
                return render_template('index1.html',prediction_text=" Expected closing price of SENSEX is: {}".format(output))
        except:
            return render_template('index1.html',prediction_text="Please,enter valid values (without commas)")
    else:
        return render_template('index1.html')

if __name__=="__main__":
    app.run(debug=True)
