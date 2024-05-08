from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Load the dataset to get tablet information
data = pd.read_csv('iris.csv')


# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('home.html')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    pulse_rate = float(request.form['pulseRate'])
    spo2 = float(request.form['spo2'])

    # Create a DataFrame with the form data
    input_data = pd.DataFrame({
        'room temp': [temperature],
        'room humidity': [humidity],
        'pulse rste': [pulse_rate],
        'spo2': [spo2]
    })

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Recommend tablet based on the predicted disease
    recommended_tablet = recommend_tablet(prediction)

    # Render the prediction template with the result
    return render_template('prediction.html',
                           prediction=prediction,
                           tablet=recommended_tablet)


def recommend_tablet(disease):
    # Find the tablet associated with the predicted disease
    tablet = data[data['disease'] == disease]['tablet'].values
    if len(tablet) > 0:
        return tablet[0]
    else:
        return 'Unknown'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
