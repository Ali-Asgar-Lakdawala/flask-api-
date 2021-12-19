from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import pandas as pd
import json


app = Flask(__name__)

modelfile = 'finalized_model.pkl'    

model = p.load(open(modelfile, 'rb'))

@app.route('/')
def main():
    return ('bike demand prediction')
    
@app.route('/api/', methods=['POST'])
def makecalc():
	j_data = request.get_json()

	prediction = np.array2string(model.predict(j_data))
	
	return jsonify(prediction,'bike demand prediction')


if __name__ == '__main__':

    
    app.run(debug=True,host='0.0.0.0')