from flask import Flask
from flask import jsonify
import numpy as np
from iris_model import trained_model

app = Flask(__name__)
version = 'v1'
model = trained_model()

@app.route('/')
def hello_world():
    response = jsonify({
        'status': 200,
        'text' : 'Hello world! '
    })
    
    response.status_code = 200
    
    return response

@app.route(f'/api/{version}/status')
def status():
    response = jsonify({
        'status': 200,
        'text' : 'The API is up and running '
    })
    response.status_code = 200
    
    return response

@app.route(f'/api/{version}/name/<string:name>')
def status_(name):
    response = jsonify({
        'status': 200,
        'text' : f'The {name}' 
    })
    response.status_code = 200
    
    return response

@app.route(f'/api/{version}/iris/<float:sl>/<float:sw>/<float:pl>/<floar:pw>')
def predict_iris(sl,sw,pl,pw):
  #prediccion del modelo
    species = ['I. setosa', 'I. versicolor','I. virginica']
    vector = np.array([[sl,sw,pl,pw]])
    prediction_vec = model.predict(vector)
    vec = [float(item) for item in prediction_vec[0]]
    prediction_class = np.argmax(prediction_vec)
    prediction_species = species(prediction_class)
        

    response = jsonify({
        'input':{
            'sepal-lenght':sl,
            'sepal-width':sw,
            'petal-lenght':pl,
            'petal-width':pw
        },
        'prediction':{
            'vector':list(prediction_vec),
            'class':int(prediction_class),
            'species':prediction_species
        }
    })
    
    response.status_code = 200
    
    return response

    

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
