# NLP Email Spam detector

## Project Overview

This project focuses on detecting spam emails using natural language processing (NLP) techniques and deep learning models. The goal is to build and compare various models to determine the most effective approach for classifying emails as either spam or ham (not spam). 

## Model Training and Evaluation

### LSTM Model

We trained an LSTM (Long Short-Term Memory) model for email spam detection. The training process was conducted three times, each with different hyperparameters to evaluate performance variations:

1. **First LSTM Model:** Initial configuration with baseline parameters.
2. **Second LSTM Model:** Modified hyperparameters to explore improvements.
3. **Third LSTM Model:** Most efficient configuration based on performance metrics.

The third iteration yielded the best results in terms of accuracy and generalization, demonstrating superior performance compared to the previous models.

### BiLSTM Model

In addition to the LSTM models, a BiLSTM (Bidirectional LSTM) model was trained. This model processes the text data in both forward and backward directions, potentially capturing more context and nuances in the email content.

### Model Comparison

For a detailed comparison of the LSTM and BiLSTM models, including performance metrics, please refer to the `models_compare.ipynb` file. This notebook provides a comprehensive analysis of both models, highlighting their strengths and weaknesses.

## GUI Application

A graphical user interface (GUI) application was developed to allow users to interact with the trained models. The application includes the following features:

- **Text File Upload:** Users can select a text file containing an email.
- **Model Selection:** Choose between the LSTM and BiLSTM models.
- **Spam/Ham Classification:** Process the email content and determine if it is spam or ham.

The GUI application can be tested by running the `GUI.py` file. The interface is designed with a baby blue color scheme and dimensions of 500x500 pixels for a user-friendly experience.

## Project Structure

- **`README.md`** - This file, providing an overview of the project.
- **`models_compare.ipynb`** - Notebook containing model comparison and analysis.
- **`GUI.py`** - Python script for the GUI application.
- **`data_preprocessing.ipynb`** - file where data is processed before training.
- **`training_LSTM.ipynb`** - file where models are trained using processed data.
- **`processing.py`** - file used in GUI to process data.


## Getting Started

1. **Install Dependencies:** Ensure you have the necessary libraries installed. You can use `pip` to install them:
   ```bash
   pip install pandas numpy matplotlib scikit-learn keras imbalanced-learn nltk
2. **Run the GUI Application:** Execute the GUI.py file to start the application.
   ```bash
   python GUI.py
## [Report]([https://website-name.com 'Link title'](https://drive.google.com/file/d/1aY9pNxmX3GnmxqlqEVM0aH0gEzv6qqpL/view?usp=sharing))

