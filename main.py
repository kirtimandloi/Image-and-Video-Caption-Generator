# Import necessary libraries
import cv2
from transformers import BartTokenizer, BartForConditionalGeneration
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from collections import OrderedDict
import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM , Embedding, Dropout, add
import tensorflow as tf
tokenizer=Tokenizer()

import pandas as pd
import cv2
import streamlit as st
from PIL import Image

import tempfile
import os

import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_summary1(text, min_length):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=min_length, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_summary(captions, min_length):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    input_text = " ".join(captions)  # Combine captions into one text
    inputs = tokenizer(input_text, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=min_length, max_length=220, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text

#
# model1=VGG16()
# model1=Model(inputs=model1.inputs,outputs=model1.layers[-2].output)
# model = tf.keras.models.load_model('best_model.h5')
# tokenizer=pd.read_pickle('tokenizer.pkl')

model1=VGG16()
model1=Model(inputs=model1.inputs,outputs=model1.layers[-2].output)
model = tf.keras.models.load_model('best_model_11.h5')
tokenizer=pd.read_pickle('tokenizer.pkl')

# Function to set background image and CSS styling
def set_background_and_style():
    # Set background image
    page_bg_img = '''
    <style>
    body {
    background-image: url('https://img.freepik.com/free-photo/abstract-digital-grid-black-background_53876-97647.jpg?size=626&ext=jpg&ga=GA1.1.1700460183.1712448000&semt=ais');
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Set CSS styling
    st.markdown(
        """
        <style>
        .title {
            color: white;
            text-align: center;
            font-size: 36px;
            padding-top: 50px;
        }
        .header {
            color: white;
            text-align: center;
            font-size: 24px;
            padding-top: 20px;
        }
        .upload {
            text-align: center;
            padding-top: 20px;
        }
        .prediction {
            color: white;
            text-align: center;
            font-size: 20px;
            padding-top: 20px;
        }
        .video-player {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image and CSS styling
set_background_and_style()

# App title and header
st.title('IMAGE AND VIDEO CAPTION GENERATOR')
st.markdown('<div class="header"> Please upload an image or a video  </div>', unsafe_allow_html=True)

# File uploader for both images and videos
file = st.file_uploader('', type=['jpeg', 'jpg', 'png', 'mp4'])

# Check if a file is uploaded
if file is not None:
    # Check if the uploaded file is an image
    if file.type.startswith('image'):
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)
        # Convert the image to array format and preprocess it
        image_array = img_to_array(image)
        resized_image_array = cv2.resize(image_array, (224, 224))
        image = img_to_array(resized_image_array)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        # Extract features using VGG16 model
        feature = model1.predict(image, verbose=0)
        # Generate caption for the image
        predicted_caption = predict_caption(model, feature, tokenizer, 35)
        predicted_caption = predicted_caption[8:-6]
        st.markdown(f'<div class="prediction"> Generated Caption: {predicted_caption} </div>', unsafe_allow_html=True)
    # Check if the uploaded file is a video
    elif file.type.startswith('video'):
        # Display the video
        st.video(file)

        # Process the video file
        # Add your video processing code here
        temp_video_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_video_path, 'wb') as f:
            f.write(file.read())

        # Process the video file
        video_capture = cv2.VideoCapture(temp_video_path)

        # video_capture = cv2.VideoCapture(file)
        frames = []
        # Extract frames from the video
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video_capture.release()
        # Process frames at the specified rate and append unique predicted captions
        captions = []
        processed_frames = 0
        i = 0
        while processed_frames < len(frames) and i < len(frames):
            print(i ,len(frames))
            frame = frames[i]
            resized_frame = cv2.resize(frame, (224, 224))
            frame_array = img_to_array(resized_frame)
            frame_array = frame_array.reshape((1, frame_array.shape[0], frame_array.shape[1], frame_array.shape[2]))
            frame_array = preprocess_input(frame_array)
            feature = model1.predict(frame_array, verbose=0)
            predicted_caption = predict_caption(model, feature, tokenizer, 35)
            predicted_caption = predicted_caption[8:-6]
            print(predicted_caption)
            if predicted_caption not in captions:
                captions.append(predicted_caption)
            processed_frames += 1
            i += 15  # Process every 3rd frame
        # Display captions for the video
        print(captions)
        summary=generate_summary(captions,80)
        print(summary)
        summary1=generate_summary1(summary,90)
        print(summary1)
        # cleaned_sentence = summary.replace('.', '')
        # original_one=generate_summary1(cleaned_sentence,40)


        # print(('\n \t' ,original_one))
        st.markdown(f'<div class="prediction"> Generated Captions for the Video: {summary1} </div>',   unsafe_allow_html=True)
