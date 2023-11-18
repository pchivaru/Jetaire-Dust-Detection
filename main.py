import streamlit as st
import pandas as pd
import numpy as np
#import importlib  
#cv2 = importlib.import_module("opencv-python-headless")
#import `opencv-python-headless' as cv2
import cv2

from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import load_img, img_to_array
from keras.models import Sequential, load_model, Model
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
#from keras import backend as K
#from keras.callbacks import ModelCheckpoint
#from keras import regularizers

from keras.applications.resnet import ResNet50, preprocess_input


def change_to_bw(bytes_data, calibration_value):
  im_gray = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE) 
  (thresh, im_bw) = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  thresh += calibration_value
  im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
  #im_bw = np.array(im_bw)

  return(im_bw, thresh)

st.title('Jetaire Dust Detection')

modelpath='./model_current_best.h5'
model = load_model(modelpath)
images_dir = './imgs/'
normed_dims = (500,500)



uploaded_file = st.file_uploader("Upload an image", type=['png','jpg'])

if uploaded_file is None:
    CALIBRATION_VALUE = -35
else:
    bytes_data = uploaded_file.read()
    CALIBRATION_VALUE= st.slider('Enter a calibration value', -70, 0, -35)

    col1, col2= st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(bytes_data)

    with col2:
        st.subheader("Processed")
        (image_bw,thres)=change_to_bw(bytes_data, CALIBRATION_VALUE)
        cv2.imwrite('./imgs/0/curr_image.jpg', image_bw)
        st.image(image_bw)

    test_datagen = ImageDataGenerator(dtype='float32',
                                preprocessing_function = preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        images_dir,
        target_size=normed_dims,
        batch_size=1,
        shuffle=False,
        #class_mode='binary'
        class_mode="categorical"
        )
    
    test_generator.reset()
    X_te, y_te = test_generator.next()

    res = model.predict(np.expand_dims(X_te[0], axis = 0))

    cat_percentages = np.array([0, 25, 50, 100])
    y_percentage = res.dot(cat_percentages)

    color = 'red'
    if y_percentage[0] < 25:
        color = 'green'

    st.header(f'Dust percentage: :{color}[{round(y_percentage[0],2)}%]')






