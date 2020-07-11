# Importing essential libraries
from flask import Flask, render_template, request
from License_Plate_Recognition_Model import Main,DetectPlates,DetectChars,PossiblePlate,PossibleChar,Preprocess
import numpy as np

# Load the Random Forest CLassifier model
#filename = 'diabetes-prediction-rfc-model.pkl'
#classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
     
        result = Main.main("images/"+image.filename)
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
	app.run(debug=True)