# import libraries
from flask import Flask, render_template, request, send_from_directory
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model
from keras.losses import SparseCategoricalCrossentropy
import os

# create flask app
app = Flask(__name__)
# load model
model = load_model('tomato.h5', custom_objects={'SparseCategoricalCrossentropy': SparseCategoricalCrossentropy})
# create upload folder
UPLOAD_FOLDER = 'static/uploads/'

# check if upload folder exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# home page
@app.route('/', methods=['GET'])
def page():
    return render_template('index.html') # return the html

# list of classes
class_labels = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
                'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
                'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# predict function
def decode_predictions(predictions):
    predicted_class = predictions.argmax(axis=-1)[0] # find index of class
    return class_labels[predicted_class], predictions[0][predicted_class] * 100 # return the class label and the confidence percentage

@app.route('/', methods=['POST'])
def predict():
    # get the inputed image and save the image
    imagefile = request.files['imagefile']
    image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
    imagefile.save(image_path)

    # pre process image
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image / 255.0

    # predict image
    yhat = model.predict(image)
    label, confidence = decode_predictions(yhat)

    # calculate the label and confidence percentage
    classification = '%s (%.2f%%)' % (label, confidence)

    # return predicted label and confidence percentage and the image
    return render_template('index.html', prediction=classification, image_path=imagefile.filename)

# send requested image from UPLOAD_FOLDER
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# run app
if __name__ == '__main__':
    app.run(port=3000, debug=True)