------------------CNNs and PyTorch Project-------------------


-----------------Project Overview----------------------------
This project focuses on training Convolutional Neural Networks (CNNs) using PyTorch. The goal is to develop and evaluate different CNN architectures on the Fashion-MNIST dataset, which includes images of various clothing items. 
The project involves implementing, training, and fine-tuning CNN models to classify these images effectively.


----------------Project Details-----------------------------
Dataset: Fashion-MNIST
Framework: PyTorch
Models: Sequential CNNs, LeNet-5
Training Methodology: The models are trained using GPU acceleration on Google Colab. Various training strategies are employed, such as using pre-trained weights and training the entire network from scratch.
Results: The project examines differences in performance based on the choice of model initialization and training strategy.


-------------Key Features------------------------------------
Sequential CNN: Developed a simple sequential CNN model with layers that are easy to modify.
LeNet-5 Implementation: Replicated the LeNet-5 architecture and applied it to the Fashion-MNIST dataset.
Training & Evaluation: Included steps to train and evaluate the models, including visualizing training and testing accuracy/loss.


----------Installation--------------
To run this project, follow these steps:
Install Dependencies: Make sure you have Python and PyTorch installed. You can install PyTorch using pip:
pip install torch torchvision matplotlib
Clone the Repository:
Clone this repository to your local machine:
git clone <repository-link>
Run the Jupyter Notebook:
Open the Jupyter Notebook file (Activity6_Divya.ipynb) and follow the instructions to run the code cells. You can execute the notebook in Google Colab or any local environment that supports Jupyter.


---------Usage-----------------------------
Select the Device: Ensure that your runtime is set to GPU if you're using Google Colab for faster training.

Train the Models: Use the provided training loops to train the CNN models on the Fashion-MNIST dataset. Different sections of the notebook will guide you through training with random weights, pre-trained weights, and testing their impact on performance.

Evaluate the Models: Visualize the performance by plotting the loss and accuracy curves. Compare results between different models and initialization strategies.



--------------Results Summary--------------
The project compares the performance of different CNNs. It shows how pre-trained weights can impact training efficiency and accuracy, and how training strategies affect the model's ability to generalize.
