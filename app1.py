#!/usr/bin/env python
# coding: utf-8

# In[93]:


from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import pandas_profiling as pp
from tqdm import tqdm
import sys
from PIL import Image, ImageDraw
from PIL import ImagePath
import urllib
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import save ,load
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, Dropout, Input, MaxPool2D , Flatten
import pickle
import random
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.metrics import iou_score
from tensorflow.keras import callbacks
import imgaug.augmenters as iaa


# In[94]:

# Defining upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Provide template folder name
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In[95]:
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


Model_path = "C:/Users/Vikrant Mohite/Desktop/Applied_AI/Case_Study-2/best_model_unet.h5"


# In[96]:


def load_model(path):

    tf.keras.backend.clear_session()
    sm.set_framework('tf.keras')
    tf.keras.backend.set_image_data_format('channels_last')
    # loading the unet model and using the resnet 50 and initialized weights with Imagenet weights.
    backbone = 'resnet50'
    IMAGE_SHAPE = (256, 1600, 3)
    model = Unet(backbone_name = backbone, input_shape = IMAGE_SHAPE, classes = 5, activation = 'softmax', \
                 encoder_freeze = True, encoder_weights = 'imagenet', decoder_block_type = 'upsampling')

    optim = tf.keras.optimizers.Adam(learning_rate= 0.001)
    focal_loss = sm.losses.cce_dice_loss
    model.compile(optim, focal_loss, metrics=[iou_score])

    model.load_weights(path)

    return model


# In[97]:


model = load_model(Model_path)


# In[35]:


classes_tocolour =   dict({0: [0, 0, 0], 1: [255, 105, 180], 2:  [180,255,105], 3:[105, 180,255], 4: [ 255, 255,105]})

def one_frame_rgb(img,classes_tocolour):
  RGB_image = []
  for outer in img :
    col = []
    for inner in outer :
      col.append(classes_tocolour.get(inner))  
    RGB_image.append(col)
  return np.array(RGB_image) #256 X 1600 X 3 


# In[101]:


# In[106]:



# In[107]:


def predict(model, images_path):
    
    image_ = images_path
        
    image = cv2.imread(image_, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)


    pred = model.predict(image)
    pred = (pred).argmax(axis = -1)
    pred = np.squeeze(pred, axis=0)
    pred = one_frame_rgb(pred, classes_tocolour)
    pred = Image.fromarray((pred*255).astype('uint8'), 'RGB')
    return pred


# In[108]:


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


# In[109]:


@app.route('/', methods=['POST', 'GET'])
def uploadFile():
    if request.method == 'POST':
        # Get the file from post request
        uploaded_img = request.files['uploaded-file']

        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename)) 

       # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # Make prediction
        predicted_image = predict(model, uploaded_image_path)
        pred_filename = 'pred1.jpg'

        predicted_image.save(os.path.join(app.config['UPLOAD_FOLDER'], pred_filename))
        session['predicted_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], pred_filename)
        return render_template('index1.html')


        

@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    img_file_path1 = session.get('predicted_img_file_path', None)

    # Display image in Flask application web page
    return render_template('show_image.html', user_image = img_file_path, predicted_image = img_file_path1)


# In[105]:


if __name__=='__main__':
    app.run(debug = True)



# In[ ]:




