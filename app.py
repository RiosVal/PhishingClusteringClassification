from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo
with open('phishing_adaboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            num_links = float(request.form['num_links'])
            num_email_addresses = float(request.form['num_email_addresses'])
            num_spelling_errors = float(request.form['num_spelling_errors'])
            num_urgent_keywords = float(request.form['num_urgent_keywords'])

            features = np.array([[num_links, num_email_addresses, num_spelling_errors, num_urgent_keywords]])
            prediction = model.predict(features)[0]

        except Exception as e:
            prediction = f'Error: {str(e)}'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)