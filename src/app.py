# app.py
from flask import Flask, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)

'''
input_data_sample ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
input_data_sample = [5.1, 3.5, 1.4, 0.2]  # y_true=0
input_data_sample = [7.0, 3.2, 4.7, 1.4]  # y_true=1
input_data_sample = [6.3, 3.3, 6.0, 2.5]  # y_true=2
'''

model_path = '/home/jovyan/iris_model/models/iris_model.pkl'
result_values = ['setosa', 'versicolor', 'virginica']

model = pickle.load(open(model_path, 'rb'))

def predict(input_data):
    input_vector = np.array(input_data).reshape((1, 4))  
    print(model.predict_proba(input_vector).argmax())
    result_idx = model.predict_proba(input_vector).argmax()
    return result_idx

def validate_range(params):
    sepal_length = float(params["sepal_length"])
    sepal_width = float(params["sepal_width"])
    petal_length = float(params["petal_length"])
    petal_width = float(params["petal_width"])
    
    if sepal_length < 4.3 or sepal_length > 7.9:
        return warning(sepal_length)
    elif sepal_width < 2.0 or sepal_width > 4.4:
        return warning(sepal_width)
    elif petal_length < 1.0 or petal_length > 6.9:
        return warning(petal_length)
    elif petal_width < 0.1 or petal_width > 2.5:
        return warning(petal_width)
    
def warning(data):
    msg = f"warning! data out of range: {data}"
    return msg

@app.route('/api/v1/info', methods=['GET'])
def get_model_info():
    info = "iris_model score 97"
    return info


@app.route('/api/v1/predict', methods=['POST'])
def predict_input_data():
    params = request.get_json()
    msg = validate_range(params)
    
    sepal_length = float(params["sepal_length"])
    sepal_width = float(params["sepal_width"])
    petal_length = float(params["petal_length"])
    petal_width = float(params["petal_width"])
    
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    print(f"get input data: {input_data}")
    
    result_idx = predict(input_data)
    result = result_values[result_idx]
    
    result_dict = {}
    result_dict["msg"] = msg
    result_dict["result"] = result
    return jsonify(result_dict)



if __name__ == '__main__':
    app.run(debug=True)