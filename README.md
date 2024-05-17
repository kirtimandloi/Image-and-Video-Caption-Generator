# Image-and-Video-Caption-Generator
Creating accurate and meaningful captions for images and videos is a difficult task because it requires a deep understanding of what is shown. This difficulty is increased by the specific challenges of using recurrent neural networks (RNNs) and dealing with the time-related aspects of videos.This guide explores the limitations of traditional RNNs, the vanishing gradient problem, and the challenges associated with CNN-LSTM models in video captioning, providing insights into potential solutions.

## Overview
The main aim of this project is to gain knowledge of deep learning techniques by implementing an image and video caption generator. We primarily use Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks for this purpose. 
The project involves merging these architectures to create a CNN-RNN model for image captioning and further integrating a Large Language Model (LLM) for enhanced video captioning.

## PROBLEM FORMULATION
### THE VANISHING GRADIENT PROBLEM AND  VIDEO CAPTIONING PROBLEM 
This is one of the most significant challenges for RNNs performance. In practice, the 
architecture of RNNs restricts its long-term memory capabilities, which are limited to only 
remembering a few sequences at a time. Consequently, the memory of RNNs is only useful for 
shorter sequences and short time-periods. 
Vanishing Gradient problem arises while training an Artificial Neural Network. This mainly 
occurs when the network parameters and hyperparameters are not properly set. The vanishing 
gradient problem restricts the memory capabilities of traditional RNNs—adding too many time-
steps increases the chance of facing a gradient problem and losing information when you use 
backpropagation.

The primary challenge with using the CNN-LSTM model to predict captions for videos lies in 
effectively handling the temporal aspect of video data. Unlike images, videos consist of 
sequential frames, and accurately capturing the temporal dynamics and context across frames 
is crucial for generating coherent and contextually relevant captions. The model needs to 
effectively encode temporal dependencies and understand the flow of events over time to 
produce accurate captions for each frame. Additionally, processing large volumes of video data 
efficiently while ensuring model scalability and performance further complicates the task of 
video captioning.

## PROPOSED WORK 
The main aim of this project is to get a little bit of knowledge of deep learning techniques. We 
use two techniques mainly CNN and LSTM for image classification. So, to make our image 
caption generator model, we will be merging these architectures. It is also called a CNN-RNN 
model. 
• CNN is used for extracting features from the image. We will use the pre-trained model Xception.
• LSTM will use the information from CNN to help generate a description of the image. 
Also for video captioning  involves integrating a Large Language Model (LLM), such as 
Facebook's   BART, into the image captioning pipeline to enhance video captioning capabilities. 
Specifically,  after generating captions for individual frames using the CNN-LSTM model, the 
captions are fed into the LLM for further processing. The LLM then performs text 
summarization on the frame captions, condensing multiple frame-level captions into a concise 
summary. This summary serves as the final caption for the video, capturing the essence of the 
entire video sequence. By leveraging the LLM's natural language processing capabilities, the 
proposed approach aims to improve the coherence, conciseness, and informativeness of video 
captions, enhancing overall video understanding and accessibility.

## Components

### CNN (Convolutional Neural Network)Purpose: 
Extract features from images.
Model Used: Xception (a pre-trained model).
LSTM (Long Short-Term Memory)Purpose: Generate descriptions for images using features extracted by the CNN.
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/9c108e25-06ea-405c-9d3a-7f5b2eb08990)

### LLM (Large Language Model)Purpose:
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/20f511a2-2417-4723-88c6-8b6b0136be1c)

Enhance video captioning by summarizing frame-level captions into a coherent video caption. 
Model Used: Facebook's BART.

## Dataset Description (Flickr8k)
The Flickr8k dataset is a public benchmark dataset designed for image-to-sentence description tasks. It consists of 8,000 images, each paired with five captions. These images are sourced from diverse groups on the Flickr website, providing a broad range of scenarios and events.

### Key Features
#### Image-Caption Pairs: 
Each image in the dataset comes with five distinct captions. 
These captions provide clear and detailed descriptions of the entities and events present in the images.
#### Diversity: 
The images depict a wide variety of events and scenarios without including well-known people or places, making the dataset more generic and suitable for broad applications.
#### Dataset Split:

  Training Set: 6,000 
  
  imagesDevelopment Set: 1,000 
  
  imagesTest Set: 1,000 images
  
### Advantages of dataset
#### Multiple Captions per Image:
Having multiple captions for each image ensures that the model can learn from different perspectives and linguistic expressions.
This variety helps in making the model more generic and reduces the risk of overfitting.
#### Diverse Categories:
The wide range of image categories in the training set helps the model generalize better to different types of images.
This diversity makes the image captioning model more robust and capable of handling various image categories effectively.

### Suitability for the Project
The Flickr8k dataset's features make it particularly suitable for developing an image caption generator. The multiple captions per image allow the model to understand and generate varied and accurate descriptions. The diverse categories of images in the dataset ensure that the model can generalize well and provide robust performance across different image types. This makes the dataset an excellent choice for training and evaluating an image captioning model, ensuring high-quality, contextually relevant captions.

### Workflow
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/eaaee03a-ddd1-4300-a7fa-f45dca754918)

#### Image Captioning: 
Use the Xception model to extract features from the input image.Pass these features to the LSTM network to generate a descriptive caption.
#### Video Captioning:
Generate captions for individual video frames using the CNN-LSTM model.Feed the frame-level captions into the BART model.
BART performs text summarization on the frame captions to create a concise summary.The summarized text serves as the final caption for the video.

## Results preview of Project
### Image Captioning 
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/3b2e792a-5231-4d09-8dec-2edde2b56a22)
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/01e28aba-692c-4e7f-9b32-b83c004353e4)
#### Image Captions look like...
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/ce9ad5f0-74ce-4b46-8169-d8d345e4865d)

### Video Captioning
![image](https://github.com/kirtimandloi/Image-and-Video-Caption-Generator/assets/159609273/ab8c5dd5-198f-4add-a188-780d6ea9b703)

## Benefits of Project
### Coherence: 
By leveraging the LLM, video captions are more coherent, effectively capturing the essence of the video sequence.
### Conciseness: 
The LLM condenses multiple frame-level captions into a single, concise summary.
### Informativeness: 
Enhanced natural language processing capabilities improve the informativeness of video captions, aiding in better video understanding and accessibility.
