from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))


@app.route('/')
def index():
    localityNames = sorted(data['localityName'].unique())
    districts = sorted(data['suburbName'].unique())
    return render_template('index.html', localityNames=localityNames, districts=districts)


@app.route('/predict', methods=['POST'])
def predict():
    localityName = request.form.get('localityName')
    bedrooms = request.form.get('bedrooms')
    size_sq_ft = request.form.get('size')
    input = pd.DataFrame([[localityName, bedrooms, size_sq_ft]], columns=[
                         'localityName', 'bedrooms', 'size_sq_ft'])
    prediction = pipe.predict(input)[0]
    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
