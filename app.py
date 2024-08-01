from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
model=pickle.load(open('DT_iris.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
  
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Create a 2D array for the model input
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    # Make the prediction
    prediction = model.predict(features)

    # Map prediction to the Iris species
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species[int(prediction[0])]
    
    return render_template('index.html', predicted_species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
