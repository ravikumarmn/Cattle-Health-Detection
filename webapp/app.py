from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('random_forest_model.joblib')
MAPPING = {"0":'Bovine Respiratory Disease',
 "1":'Bovine Viral Diarrhea',
 "2":'Foot and Mouth Disease',
 "3":'Bovine Tuberculosis',
 "4":'Ringworm',
 "5":'Healthy',
 "6":'Brucellosis',
 "7":'Anthrax',
 "8":'Leptospirosis',
 "9":"Johne's Disease",
 "10":'Salmonella',
 "11":'Mastitis',
 "12":'Infectious Bovine Rhinotracheitis',
 "13": 'Blue Tongue'}

# Define the home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Extract the input values from the form
        age = request.form['age']
        weight = request.form['weight']
        body_condition_score = request.form['body_condition_score']
        temperature = request.form['temperature']
        heart_rate = request.form['heart_rate']
        milk_yield = request.form['milk_yield']

        # Create a dataframe with the input values
        input_df = pd.DataFrame(
            [[age, weight, body_condition_score, temperature, heart_rate, milk_yield]],
            columns=['age', 'weight', 'body_condition_score', 'temperature', 'heart_rate', 'milk_yield'])
        # Make a prediction using the model
        prediction = model.predict(input_df)
        prediction = MAPPING[str(int(prediction))]
        # Return the prediction as a string
        return render_template("index.html", prediction=prediction)

    # Return the home page
    return render_template('index.html')

# Define the visualize page route
@app.route('/visualize')
def visualize():
    # Render the visualize.html template
    return render_template('visualize.html')


if __name__ == '__main__':
    app.run(debug=True)
