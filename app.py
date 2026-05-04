from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load Model dan KEDUA Scaler
model = load_model('model_jst.h5')
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.form.getlist('noise_input[]')
        inputs = np.array([float(x) for x in raw_data]).reshape(1, -1)
        
        # Gunakan scaler_X untuk input (5 fitur)
        inputs_scaled = scaler_X.transform(inputs)
        
        pred_scaled = model.predict(inputs_scaled)
        
        # Gunakan scaler_y untuk hasil (1 fitur)
        prediction = scaler_y.inverse_transform(pred_scaled)
        
        final_val = round(float(prediction[0][0]), 2)
        return jsonify({'prediction': final_val})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)