# Spotify User Reviews

## Overview
The Spotify User Reviews project processes, analyzes, and models Spotify user feedback. Using NLP and machine learning, it classifies user sentiment and extracts insights. The workflow includes data preprocessing, model training, evaluation, and deployment.

## Project Structure

### Data Files (`data/`)
Contains raw datasets with Spotify user reviews for preprocessing and analysis.

### Data Processing (`data_processing/`)
Scripts to:
- **Clean and preprocess text data** for analysis.
- **Feature engineering** to extract relevant features for ML modeling.

### Model Training and Evaluation (`model/model_evaluation.py`)
This directory contains scripts for training and evaluating machine learning models:
- **Model Training**: Trains several classifiers on processed data.
- **MLflow Tracking**: Tracks experiments, storing metadata and metrics (accuracy, F1 score) for each model configuration.
- **Hyperparameter Tuning with Optuna**: Optimizes model hyperparameters to improve performance through automated testing of configurations.

### ML Artifacts (`mlartifacts/`)
Stores the output from training, including:
- **Trained Models**: Best-performing models saved after training.
- **Experiment Results**: Saved results from MLflow for analysis and comparison.

### Text Preprocessing Module (`text_preprocessor_package/`)
A package for:
- **Tokenization, lemmatization, stemming**, and other NLP tasks.
- Prepares data for feature extraction in ML pipelines.

### Dataset Analysis Notebook (`dataset_analysis.ipynb`)
An exploratory notebook that:
- Analyzes dataset structure, sentiment distribution, and trends.
- Visualizes insights from user review data.

### Model Deployment with FastAPI (`main.py`)
The main script deploys the trained model using [FastAPI](https://fastapi.tiangolo.com/). The API:
- Accepts user input as text reviews.
- Processes and classifies the text input, returning sentiment predictions.
- Enables real-time interaction with the trained sentiment analysis model.

### Dockerfile
Defines a Docker environment for easy deployment.

### Requirements (`requirements.txt`)
Lists all dependencies for consistent project setup.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/grzegorzlucki/Spotify-User-Reviews.git
   cd Spotify-User-Reviews
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the project:**
   ```bash
   python main.py
4. **Docker setup (optional):**
   ```bash
   docker build -t spotify-user-reviews .
   docker run spotify-user-reviews
