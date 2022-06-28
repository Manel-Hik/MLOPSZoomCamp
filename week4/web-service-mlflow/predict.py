import pickle
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)



def prepare_features(ride):
    features = {}
    features['PU_DO'] =  '%s_%s' % (ride['PUlocationID'] ,ride["DOlocationID"])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]



app = Flask('Duration-Prediction')

#this decorator to turn our function into http request
@app.route('/predict', methods = ['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)