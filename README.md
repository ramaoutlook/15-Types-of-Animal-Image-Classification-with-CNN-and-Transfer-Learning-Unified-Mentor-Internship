# Animal Image Classification

This project implements an animal image classification model using transfer learning with TensorFlow and Keras in Google Colab. The goal is to classify images of various animals from a dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Visualizations](#visualizations)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

The project focuses on building and training a deep learning model to identify 15 different animal species. It utilizes transfer learning with pre-trained models (MobileNetV2, VGG16, or ResNet50) to leverage their learned features and achieve better performance with a smaller dataset and less training time. The entire workflow is designed to run seamlessly in Google Colab.

## Dataset
Below is the link for the dataset which is used in this project.
https://drive.google.com/file/d/1wPyI-x_cqGrz3x2ecR7OdS1aTow2DdBG/view?usp=drive_link
The dataset used for training and validation contains images of the following animal classes:
- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

The dataset is expected to be organized in a directory structure where each subdirectory corresponds to a class name and contains the images for that class. The code includes methods to load the dataset from a local upload or from Google Drive (including unzipping a zip file).

## Model Architecture

The model employs transfer learning using a pre-trained convolutional base and a custom classification head.

- **Base Model:** The code allows selection from MobileNetV2 (recommended for Colab), VGG16, or ResNet50. The pre-trained weights from the ImageNet dataset are used. The convolutional base layers are frozen during initial training to preserve the learned features.
- **Classification Head:** A custom head is added on top of the base model, consisting of:
    - Global Average Pooling 2D layer
    - Dropout layers to prevent overfitting
    - Dense layers with ReLU activation
    - A final Dense layer with softmax activation for multi-class classification

The model is compiled with the Adam optimizer and categorical crossentropy loss.

## Training

The model is trained using data augmentation techniques to increase the diversity of the training data and improve generalization. The training process includes:

- **Data Augmentation:** Techniques like rotation, shifting, shearing, zooming, and horizontal flipping are applied to the training images.
- **Callbacks:** Early Stopping, ReduceLROnPlateau, and Model Checkpointing are used to monitor training progress, adjust the learning rate, and save the best performing model.
- **Epochs and Batch Size:** The training is performed for a defined number of epochs with a specified batch size.

## Visualizations

The project includes visualizations to track the training progress and evaluate the model's performance.

### Accuracy and Loss Plots

These plots show the training and validation accuracy and loss over epochs, helping to identify overfitting or underfitting.

![Image](https://github.com/user-attachments/assets/c8383b80-a0ce-41a8-a0ab-70086172037e)


### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's predictions, showing true positives, true negatives, false positives, and false negatives for each class.

![Image](https://github.com/user-attachments/assets/5aa061e8-c6e5-4e59-aec8-5e0ec19026b9)


## Evaluation

The model's performance is evaluated on the validation set using metrics such as:

- **Validation Accuracy**
- **Validation Loss**
- **Classification Report:** Includes precision, recall, F1-score, and support for each class.
- **Confusion Matrix**

## Prediction

A function is provided to make predictions on a single image. It displays the image, the predicted class, and the confidence score, along with the top 3 predictions.

### Sample Prediction Output

![Image](https://github.com/user-attachments/assets/f63bf049-3e9c-4356-939f-c4627ede1df4)


## Getting Started

1. **Open in Google Colab:** Open the notebook in Google Colab.
2. **Connect to GPU:** Ensure you are connected to a GPU runtime for faster training (Runtime -> Change runtime type -> GPU).
3. **Mount Google Drive (Optional):** If your dataset is in Google Drive, mount your Drive using the provided code cell.
4. **Upload or Specify Dataset Path:**
   - **Method 1 (Upload):** Use the `upload_dataset_zip()` function (if available in your code) to upload a zip file directly.
   - **Method 2 (Google Drive):** Update the `drive_dataset_path` variable in the `use_drive_dataset()` function to the path of your dataset (zip file or directory) in Google Drive. Uncomment the line `DATASET_PATH = use_drive_dataset()`.
5. **Run Cells:** Execute the cells in order. The notebook will guide you through setup, data loading, model building, training, evaluation, and saving the model.
6. **Test Prediction:** Use the `test_prediction_colab()` function with the path to a test image to see a sample prediction.

## Dependencies

The project requires the following libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- `google.colab` (for Colab-specific functionalities)

These dependencies are typically pre-installed in Google Colab.

## License
All rights are reserved by me.

