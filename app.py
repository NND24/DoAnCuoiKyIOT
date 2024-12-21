from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import json
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load attack features from JSON
ATTACK_FEATURES_FILE = './0004-Performance-Evaluation/GA_output_ET.json'
def load_attack_features(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)
attack_features = load_attack_features(ATTACK_FEATURES_FILE)

# Model paths
model_paths = {
    'DT': {
        'SYN': './0004-Performance-Evaluation/models/DT_SYN_1_model.pkl',
        'HTTP': './0004-Performance-Evaluation/models/DT_HTTP_1_model.pkl',
        'ACK': './0004-Performance-Evaluation/models/DT_ACK_1_model.pkl',
        'UDP': './0004-Performance-Evaluation/models/DT_UDP_1_model.pkl',
        'ARP': './0004-Performance-Evaluation/models/DT_ARP_1_model.pkl',
        'SP': './0004-Performance-Evaluation/models/DT_SP_1_model.pkl',
        'BF': './0004-Performance-Evaluation/models/DT_BF_1_model.pkl',
        'OS': './0004-Performance-Evaluation/models/DT_OS_1_model.pkl',
        'SCHD': './0004-Performance-Evaluation/models/DT_SCHD_1_model.pkl',
        'MHDis': './0004-Performance-Evaluation/models/DT_MHDis_1_model.pkl'
    },
    'LR': {
        'SYN': './0004-Performance-Evaluation/models/LR_SYN_1_model.pkl',
        'HTTP': './0004-Performance-Evaluation/models/LR_HTTP_1_model.pkl',
        'ACK': './0004-Performance-Evaluation/models/LR_ACK_1_model.pkl',
        'UDP': './0004-Performance-Evaluation/models/LR_UDP_1_model.pkl',
        'ARP': './0004-Performance-Evaluation/models/LR_ARP_1_model.pkl',
        'SP': './0004-Performance-Evaluation/models/LR_SP_1_model.pkl',
        'BF': './0004-Performance-Evaluation/models/LR_BF_1_model.pkl',
        'OS': './0004-Performance-Evaluation/models/LR_OS_1_model.pkl',
        'SCHD': './0004-Performance-Evaluation/models/LR_SCHD_1_model.pkl',
        'MHDis': './0004-Performance-Evaluation/models/LR_MHDis_1_model.pkl'
    }
}

def predict_attack(input_data, selected_algorithm):
    final_results = []

    for attack, model_path in model_paths[selected_algorithm].items():
        model = joblib.load(model_path)
        features = attack_features[attack]
        filtered_data = input_data[features[:-1]]  # Exclude 'Label'
        probabilities = model.predict_proba(filtered_data)[:, 1]
        max_prob = max(probabilities)

        if max_prob > 0.5:
            final_results.append({"Attack_Type": attack, "Probability": max_prob})
    
    if final_results:
        return sorted(final_results, key=lambda x: x['Probability'], reverse=True)[:1]
    return []

def predict_from_csv(file_path, selected_algorithm, output_file):
    input_data = pd.read_csv(file_path)
    results = []

    for index, row in input_data.iterrows():
        prediction = predict_attack(pd.DataFrame([row]), selected_algorithm)
        if prediction:
            results.append({"Index": index, "Attack_Type": prediction[0]["Attack_Type"], "Probability": prediction[0]["Probability"]})
        else:
            results.append({"Index": index, "Attack_Type": "None", "Probability": 0})
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    return result_df

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        algorithm = request.form.get('algorithm')

        if file and algorithm:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            output_file = os.path.join("uploads", f"result_{file.filename}")
            result_df = predict_from_csv(file_path, algorithm, output_file)

            # Render results to the template
            results = result_df.to_dict('records')
            return render_template("index.html", results=results, algorithm=algorithm, output_file=output_file)

    return render_template("index.html", results=None)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
