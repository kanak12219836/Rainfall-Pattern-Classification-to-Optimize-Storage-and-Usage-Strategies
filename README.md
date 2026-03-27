
 Rainfall Pattern Classification to Optimize Storage and Usage Strategies

 Overview

This project focuses on classifying rainfall patterns using deep learning to support better water storage and usage strategies. The model analyzes rainfall images and categorizes them into three classes: heavy, medium, and light rainfall.

 Dataset

* Training images: 748
* Validation images: 73
* Classes: Heavy, Medium, Light

Preprocessing techniques include histogram equalization and data augmentation (rotation, zoom, shift, and flip).

 Approach

The project follows these steps:

* Dataset preparation and validation split
* Image resizing to 224×224
* Data normalization and augmentation
* Model training using deep learning techniques

 Models Used

* Custom CNN
* ResNet50 (Transfer Learning)
* Xception (Transfer Learning)

 Results

The Xception model performed best with:

* Training Accuracy: 81%
* Validation Accuracy: 89%

 Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Matplotlib
* PIL


 Future Work

* Improve dataset size and diversity
* Deploy as a web application
* Enable real-time rainfall detection


