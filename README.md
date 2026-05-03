# Fake News Detector

This is a web application built with Streamlit that detects fake news articles using machine learning. The app analyzes news titles and bodies to classify them as real or fake news, providing predictions along with feature explanations.

## Features

- **News Classification**: Input a news title and body to get a prediction (Real or Fake).
- **Feature Explanations**: Understand why the model made its decision by viewing the top contributing features.
- **Interactive UI**: Simple and user-friendly interface powered by Streamlit.

## Installation

1. **Clone the repository** (if applicable) or ensure you have the project files in your local directory.

2. **Create a virtual environment** (recommended):
   ```
   python -m venv env
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```
     env\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source env/bin/activate
     ```

4. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the model files (`model.pkl` and `tfidf.pkl`) are in the same directory as `app.py`. These are generated from the training notebook.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Enter a news title and body in the input fields and click "Predict" to see the results.

## Files

- `app.py`: The main Streamlit application.
- `train.ipynb`: Jupyter notebook for training the model.
- `WELFake_Dataset.csv`: The dataset used for training.
- `requirements.txt`: List of Python dependencies.

## Dataset

The dataset used for training is the WELFake dataset. You can find it here: [Dataset Link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data?select=WELFake_Dataset.csv)