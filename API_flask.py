from flask import Flask, render_template, request, jsonify, make_response
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    print('start')
    return render_template('digit.html')

@app.route('/hi', methods=['GET'])
def sayhi():
    return jsonify({'result':'ok'})

@app.route('/add', methods=['POST'])
def add():
    request_data_a = request.values['a']
    request_data_b = request.values['b']
    result = int(request_data_a) + int(request_data_b)
    return jsonify({'result': result})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file.save('digit.png')
    img = load_img('digit.png')
    data = img_to_array(img)
    R, G, B = data[:,:,0], data[:,:,1], data[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    model = load_model('MyMinistModel.h5')
    imgGray = imgGray.reshape(1,28,28)
    imgGray = imgGray.reshape(1,28,28,1)
    ret = np.argmax(model.predict(imgGray),axis=-1)
    return render_template('result.html', digitvalue=ret[0])


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)