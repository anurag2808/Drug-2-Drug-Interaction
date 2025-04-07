from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # ✅ Allows cross-origin requests

# Load the dataset
csv_file = "/Users/ishabisen/Documents/SEM VI/ML/ML PROJECT/FINAL WORKING PROJECT/Id_details_new.csv"
df = pd.read_csv(csv_file)

# Load the trained model
model_file = "/Users/ishabisen/Documents/SEM VI/ML/ML PROJECT/FINAL WORKING PROJECT/random_forest_model_with_sampling.pkl"
model = joblib.load(model_file)

# Function to get the feature values for given IDs
def get_features(id_1, id_2):
    row1 = df[df['ID'] == id_1]
    row2 = df[df['ID'] == id_2]

    if row1.empty or row2.empty:
        return None

    features = {
        'MolWt_X1': row1['MolWt_X'].values[0],
        'LogP_X1': row1['LogP_X'].values[0],
        'NumHDonors_X1': row1['NumHDonors_X'].values[0],
        'NumHAcceptors_X1': row1['NumHAcceptors_X'].values[0],
        'TPSA_X1': row1['TPSA_X'].values[0],
        'MolWt_X2': row2['MolWt_X'].values[0],
        'LogP_X2': row2['LogP_X'].values[0],
        'NumHDonors_X2': row2['NumHDonors_X'].values[0],
        'NumHAcceptors_X2': row2['NumHAcceptors_X'].values[0],
        'TPSA_X2': row2['TPSA_X'].values[0],
    }

    return pd.DataFrame([features])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received Data:", data)

    id_1 = data.get("id1")
    id_2 = data.get("id2")

    features_df = get_features(id_1, id_2)
    if features_df is None:
        return jsonify({"error": "One or both DrugBank IDs not found"}), 400

    print("Extracted Features:", features_df)

    prediction = model.predict(features_df)
    print("Prediction:", prediction.tolist())

    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777, debug=True)