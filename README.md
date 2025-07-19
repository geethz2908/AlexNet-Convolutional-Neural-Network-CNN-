📌 Project Overview
This project implements the classic AlexNet Convolutional Neural Network (CNN) architecture from scratch using TensorFlow and Keras on the CIFAR-10 dataset for image classification. It demonstrates key deep learning concepts such as model building, training, evaluation, and visualization.

🎯 Objectives
Understand and implement the AlexNet CNN architecture

Preprocess and augment the CIFAR-10 dataset

Train and evaluate the model using standard metrics

Visualize training metrics (accuracy, loss) using TensorBoard

Make and visualize predictions on sample test images

🛠️ Tools & Technologies
Language: Python

Frameworks/Libraries: TensorFlow, Keras, NumPy, Matplotlib

Dataset: CIFAR-10

Visualization: TensorBoard

🧠 About AlexNet
Developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, AlexNet is an 8-layer deep CNN that revolutionized image recognition.

Key Features:
ReLU activation for faster convergence

Dropout regularization to prevent overfitting

Data augmentation (translations and reflections)

MaxPooling and Batch Normalization

GPU-accelerated training

📚 Dataset: CIFAR-10
60,000 color images (32x32 pixels)

10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

50,000 images for training and 10,000 for testing

Challenges: low resolution, class imbalance, complex backgrounds

🧪 Preprocessing
Images resized to 64x64 pixels

Efficient loading with tf.data.Dataset

Normalization and data shuffling

Optional data augmentation

🏗️ Model Architecture
Custom AlexNet design including:

5 Convolutional Layers: with ReLU, Batch Normalization, and MaxPooling

3 Fully Connected Layers: with Dropout and ReLU

Output Layer: Softmax activation for 10-class classification

⚙️ Training Details
Optimizer: SGD

Learning Rate: 0.001

Loss Function: Sparse Categorical Crossentropy

Epochs: 20

Evaluation Metric: Accuracy

📈 Results & Evaluation
Plots of training and validation accuracy/loss

Sample predictions vs. actual labels

Final test set accuracy reported

📊 TensorBoard Visualization
Real-time visualization of:

Accuracy and loss curves

Computational graph of the model

🖼️ Sample Outputs
Training history plots

Image classification results with predicted vs. actual labels

