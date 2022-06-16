import joblib
import numpy
from flask import Flask, request, Response

MODEL_PATH_V1 = 'mlmodels/model_version_1.pkl'
SCALER_X_PATH_V1 = 'mlmodels/scaler_x_version_1.pkl'
SCALER_Y_PATH_V1 = 'mlmodels/scaler_y_version_1.pkl'

MODEL_PATH_V2 = 'mlmodels/model_version_2.pkl'
SCALER_X_PATH_V2 = 'mlmodels/scaler_x_version_2.pkl'
SCALER_Y_PATH_V2 = 'mlmodels/scaler_y_version_2.pkl'

app = Flask(__name__)

model_v1 = joblib.load(MODEL_PATH_V1)
sc_x_v1 = joblib.load(SCALER_X_PATH_V1)
sc_y_v1 = joblib.load(SCALER_Y_PATH_V1)

model_v2 = joblib.load(MODEL_PATH_V2)
sc_x_v2 = joblib.load(SCALER_X_PATH_V2)
sc_y_v2 = joblib.load(SCALER_Y_PATH_V2)


@app.route("/predict_price", methods=['GET'])
def predict():
    args = request.args
    model_version = args.get('model_version', default=-1, type=int)
    floor = args.get('floor', default=-1, type=int)
    open_plan = args.get('open_plan', default=-1, type=int)
    rooms = args.get('rooms', default=-1, type=int)
    studio = args.get('studio', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    kitchen_area = args.get('kitchen_area', default=-1, type=float)
    living_area = args.get('living_area', default=-1, type=float)
    renovation = args.get('renovation', default=-1, type=int)

    if model_version == 1:
        parameters = [open_plan, rooms, area, renovation]
        if any([i == -1 for i in parameters]):
            return Response(status=500)
        x = numpy.array([open_plan, rooms, area, renovation]).reshape(1, -1)
        x = sc_x_v1.transform(x)
        result = model_v1.predict(x)
        result = sc_y_v1.inverse_transform(result.reshape(1, -1))
        return str(result[0][0])

    elif model_version == 2:
        parameters = [floor, open_plan, rooms, studio, area, kitchen_area, living_area, renovation]
        if any([i == -1 for i in parameters]):
            return Response(status=500)
        x = numpy.array([floor, open_plan, rooms, studio, area, kitchen_area, living_area, renovation]).reshape(1, -1)
        x = sc_x_v2.transform(x)
        result = model_v2.predict(x)
        result = sc_y_v2.inverse_transform(result.reshape(1, -1))
        return str(result[0][0])

    else:
        return Response(status=500)


if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')
