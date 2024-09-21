# Handwritten Digit Recognition using CNN and Streamlit

### Overview
This project is a Handwritten Digit Recognition application built using a Convolutional Neural Network (CNN) and deployed with Streamlit for real-time user interaction. The model is trained on the MNIST dataset and can recognize digits from user-uploaded images.

###  Features
- CNN Model: Built using TensorFlow and trained on the MNIST dataset.

- Streamlit Deployment: Allows users to upload their own handwritten digits and get predictions in real-time.

- Data Augmentation: Enhances the model’s ability to generalize on unseen data by using techniques like random rotations and shifts.

1. Upload an Image: You can upload a 28x28 grayscale image of a handwritten digit. It’s recommended to use a bold, thick font for better accuracy.
2. Get Prediction: Once the image is uploaded, the app will predict the digit based on the trained CNN model.

### Example Image Format
- Resolution: 28x28 pixels

- Color: Grayscale

- Font Recommendation: To get accurate predictions, it is recommended to use a bold, thick font when writing digits.
Dataset

- The model was trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

### Project Structure
- app.py: Main Streamlit app script.

- model.h5: Trained CNN model.

- model.ipynb: CNN model full code

- README.md: Project documentation.


### Technology Stack
- TensorFlow: For building and training the CNN model.

- Streamlit: For deploying the model in a web application.

- NumPy, Pandas, Matplotlib: For data manipulation and visualization.


### Future Enhancements
- Improve model accuracy on images drawn with different fonts or styles.
- Add functionality to process colored or larger resolution images.

