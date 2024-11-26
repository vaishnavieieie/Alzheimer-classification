from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask('ping')
# Routes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load models
        model1 = pickle.load(open('random_forest_model.pkl', 'rb'))
        model2 = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
        model3 = pickle.load(open('adaboost_model.pkl', 'rb'))
        model4 = pickle.load(open('random_forest_best_model.pkl', 'rb'))
        # Get the data from the request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert the data into a DataFrame
        data_df = pd.DataFrame(data, index=[0])


        # Predict with all models
        result1 = model1.predict(data_df)
        result2 = model2.predict(data_df)
        result3 = model3.predict(data_df)
        result4 = model4.predict(data_df)
        # Predicting probability
        prob1 = model1.predict_proba(data_df)
        prob2 = model2.predict_proba(data_df)
        prob3 = model3.predict_proba(data_df)
        prob4 = model4.predict_proba(data_df)
        print("hello",prob1)

        # print(result1)
        # Return the predictions
        return jsonify({
            "RandomForest": {
                "pred": int(result1[0]),
                "prob": prob1[0][1],
                },
            "GradientBoosting": {
                "pred": int(result2[0]),
                "prob": prob2[0][1],},
            "AdaBoost": {
                "pred": int(result3[0]),
                "prob": prob3[0][1],},
            "RandomForestBest": {
                "pred": int(result4[0]),
                "prob": prob4[0][1]}
        
                
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
