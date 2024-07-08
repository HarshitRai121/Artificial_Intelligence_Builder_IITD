# Capstone Project: Emotion Detection System

## Overview
1. [Introduction](#introduction)
2. [Project Plan](#project-plan)
3. [Objectives](#objectives)
4. [Methodology](#methodology)
5. [Review of the Models Used](#review-of-the-models-used)
6. [Experimental Results](#experimental-results)
7. [Conclusion](#conclusion)

## Introduction
The project aimed to develop an Emotion Detection System using deep learning to identify and categorize facial expressions into different emotions.

### Background
Understanding human emotions is crucial for intelligent systems that interact with users. Facial expressions are key indicators of emotions, making them valuable for emotion detection systems. This project uses Convolutional Neural Networks (CNNs) to extract features from facial images and classify them into predefined emotion categories.

### Objectives
- Develop a CNN-based model to recognize facial expressions accurately.
- Implement real-time emotion detection in live video feeds.
- Create a user interface for both image and video emotion detection.

### Scope
The project covers the entire emotion detection pipeline, from data preprocessing and model training to real-time application. It uses TensorFlow and Keras for deep learning, OpenCV for image/video processing, and Tkinter for the GUI.

## Project Plan
### Contribution by Team Members
1. Preprocess train images: Harshit Rai
2. Create CNN model structure: Godavari Patle
3. Train the neural network model: Rohit Ahirwar
4. Accuracy and loss evaluation: Bhagwati Ahirwar
5. GUI: Sakshi Khatarkar

### Schedule
| Phase                          | Tasks                               | Start | Duration (Days) |
|--------------------------------|-------------------------------------|-------|-----------------|
| Literature Survey              | Identify relevant research          | 1     | 1               |
|                                | Read and analyze the papers         | 2     | 2               |
|                                | Summary of Literature Survey        | 4     | 1               |
| Data Collection and Preprocessing | Data Collection                   | 5     | 1               |
|                                | Exploring the data                  | 6     | 1               |
| Model Training and Evaluation  | Create model architecture           | 7     | 1               |
|                                | Training the model                  | 8     | 2               |
|                                | Plotting loss graphs and analysis   | 10    | 1               |
| Novel Architecture Development | Design GUI                          | 11    | 2               |
| Documentation                  | Draft report and PPT preparation    | 13    | 2               |
|                                | Final report and PPT preparation    | 15    | 2               |
| **Total Duration**             |                                     |       | 16 (Days)       |

## Objectives
- Develop an accurate emotion detection model using CNNs.
- Train and validate the model using the FER 2013 dataset.
- Create an intuitive GUI for easy uploading and analysis of images.
- Contribute to affective computing by providing a tool for applications like mental health analysis, customer feedback systems, and interactive entertainment.

## Methodology
### Dataset Details
- **Origin:** FER 2013 dataset from ICML 2013 Challenges in Representation Learning.
- **Content:** Grayscale images of human faces labeled with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Total Images:** 35,887; Training Set: 28,709; Public Test Set: 3,589; Private Test Set: 3,589.
- **Challenges:** Real-world, uncontrolled facial expressions.

### Data Preprocessing and Augmentation
- **Techniques:** Grayscale conversion, resizing, histogram equalization, horizontal flipping, random rotation, zooming, data normalization.
- **Objective:** Enhance image data for better model performance and increase dataset diversity without new images.

### Model Training
- **Architecture:** CNN tailored for facial expression recognition.
- **Hyperparameters:** Learning rate, batch size.
- **Callbacks:** Model checkpointing, reduce learning rate on plateau.
- **Evaluation:** Accuracy, loss, validation steps, performance visualization.

## Review of the Models Used
- **Architecture:** Multiple convolutional layers followed by dense layers, ELU activation, dropout, and batch normalization.
- **Evolution:** Started with a basic CNN, improved with more layers, ELU activation, dropout, and learning rate adjustments.
- **Performance:** Training accuracy improved from 31.43% to 66.92%; validation accuracy from 41.43% to 63.45%.

## Experimental Results
| Layer Type         | Configuration                                    | Details                                       |
|--------------------|--------------------------------------------------|-----------------------------------------------|
| Input              | 48x48x1                                          | Grayscale images                              |
| Conv Layer 1       | 64 filters, 3x3 kernel, padding="same", ELU      | MaxPooling2D, Dropout (0.25)                  |
| Conv Layer 2       | 128 filters, 5x5 kernel, padding="same", ELU     | MaxPooling2D, Dropout (0.25)                  |
| Conv Layer 3       | 256 filters, 3x3 kernel, padding="same", ELU     | MaxPooling2D, Dropout (0.25)                  |
| Conv Layer 4       | 512 filters, 3x3 kernel, padding="same", ELU     | MaxPooling2D, Dropout (0.25)                  |
| Conv Layer 5       | 512 filters, 3x3 kernel, padding="same", ELU     | MaxPooling2D, Dropout (0.25)                  |
| Flatten            | -                                                | Converts 2D matrix data to a vector           |
| Dense 1            | 256 nodes, ELU                                   | Dropout (0.25)                                |
| Dense 2            | 512 nodes, ELU                                   | Dropout (0.25)                                |
| Output             | 7 nodes (Softmax)                                | Corresponds to the 7 emotion classes          |

### Training Details
- **Optimizer:** Adam, Learning rate: 0.0005.
- **Loss Function:** Categorical Crossentropy.
- **Metrics:** Accuracy.
- **Epochs:** 30.
- **Batch Size:** 64.
- **Callbacks:** ModelCheckpoint, ReduceLROnPlateau.

## Conclusion
- **Effective Model:** Developed a CNN model with convolutional layers, ELU activations, pooling, and dropout techniques, achieving a training accuracy of 66.92% and validation accuracy of 63.45%.
- **Real-world Applications:** Potential uses include enhancing user experience, aiding psychological studies, and improving human-computer interaction.
- **Challenges Addressed:** Overcame issues like dataset imbalance and real-time processing.
- **Future Enhancements:** Plan to explore deeper architectures, larger datasets, and real-time video emotion detection for improved accuracy and functionality.

