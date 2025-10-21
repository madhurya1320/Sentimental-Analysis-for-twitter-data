from keras.models import load_model
from flask import Flask,request,jsonify,render_template
from sklearn.feature_extraction.text import CountVectorizer
import pickle
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    with open('cf19.pkl','rb') as file:
        cv=pickle.load(file)    
        model=load_model('final3.h5')
        prediction = model.predict(cv.transform([request.form['Sentence']]))
        output=prediction[0]
        if(output<0.5):
            return render_template('index.html',prediction_text="The tweet has negative expressions")
        else:
            return render_template('index.html',prediction_text="The tweet has Positive expressions",image=True)
if(__name__=="__main__"):
    app.run(debug=True)
