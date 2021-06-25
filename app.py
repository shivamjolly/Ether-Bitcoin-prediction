import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('xgb_modeln.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features =[ [float(x) for x in request.form.values()]]
    final_features = np.array(int_features)
    print(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=' Predicted etheruem price is $ {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)