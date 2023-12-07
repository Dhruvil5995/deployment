import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

model = load_model('brain_tumor.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(file_path):
    IMAGE_SIZE = 64

    # Load and preprocess the image
    image = cv2.imread(file_path)
    img = Image.fromarray(image)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)
    final_img = np.expand_dims(img, axis=0)

    # Perform the prediction
    result = model.predict(final_img)
    result = np.round(result).astype(int)[0][0]

    # Return the prediction result
    if result == 1:
        return 'Detect brain tumor'
    else:
        return 'No brain tumor detected'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file and allowed_file(image_file.filename):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(image_path)
            image_url = url_for('static', filename='uploads/' + image_file.filename)
            prediction = predict_image(image_path)
            return render_template('index.html', prediction=prediction, image=image_url)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)



