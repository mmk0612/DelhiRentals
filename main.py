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
    try:
        localityName = request.form.get('localityName')
        bedrooms = int(request.form.get('bedrooms'))
        size_sq_ft = float(request.form.get('size'))

        # Validate input values if needed
        if bedrooms < 0 or size_sq_ft <= 0:
            return jsonify({'error': 'Invalid input values'}), 400

        input_data = pd.DataFrame({'localityName': [localityName],
                                   'bedrooms': [bedrooms],
                                   'size_sq_ft': [size_sq_ft]})

        prediction = pipe.predict(input_data)[0]
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle unexpected errors


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
