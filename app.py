# Importing essential libraries
from flask import Flask, render_template, request
from preprocess import preprocess
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'nlp_model1.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        message = request.form['message']
        b=preprocess(message)
        data = [b]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
