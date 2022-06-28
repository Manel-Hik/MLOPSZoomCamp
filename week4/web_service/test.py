import requests
# import predict
ride = {
    "PUlocationID":10,
    "DOlocationID":50,
    "trip_distance":40
    }

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url,json = ride)
print(response.json())
# features = predict.prepare_features(ride)
# pred = predict.predict(features)
# print(pred)
#docker build -t ride-duration-prediction-service:v1 .
# test it
#docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1

#it: run it into an interactive mode
#--rm remove the image after done with this 
