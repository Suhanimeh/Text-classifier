# NLP Text Classifier

**A modular, command-line text classification pipeline built with NLTK and scikit-learn.**

This project uses Natural Language Processing (NLP) and machine learning to classify news headlines and articles into predefined categories, such as Sports, Technology, Politics, and Entertainment.

The pipeline supports text preprocessing, model training, evaluation, and single-text or batch predictions through a command-line interface (CLI). Its modular structure makes it easier to understand, test, and extend individual stages of the machine-learning workflow.

## Project Objectives

- Build a reusable NLP pipeline for news text classification.
- Clean and preprocess text before model training.
- Compare multiple machine-learning algorithms.
- Evaluate classification performance using standard metrics.
- Support predictions for individual texts and batches of text.
- Organise the codebase into modular components for testing and maintenance.

## Features

- **Text Preprocessing:** Clean and normalise text using NLP techniques.
- **Feature Extraction:** Convert text into numerical features suitable for machine-learning models.
- **Multiple Classification Models:** Support Logistic Regression, Naive Bayes, Support Vector Machines, and Random Forest, provided these options are implemented in the training pipeline.
- **Model Training:** Train a classifier using a labelled news dataset.
- **Model Evaluation:** Generate classification reports and assess model performance.
- **Single-Text Prediction:** Predict the category of an individual headline or article.
- **Batch Prediction:** Process multiple text records from an input file.
- **Command-Line Interface:** Run training, evaluation, and prediction tasks without using a notebook.
- **Automated Testing:** Test pipeline components using the project's test suite.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Implements the classification pipeline |
| NLP | NLTK | Supports text-processing operations |
| Machine Learning | scikit-learn | Provides classification algorithms and evaluation tools |
| Data Processing | pandas, NumPy | Supports data handling and numerical operations |
| Model Persistence | joblib | Saves and loads trained models |
| CLI | Click | Builds command-line commands and options |
| Testing | pytest | Runs automated tests |
| Exploration | Jupyter Notebook | Supports exploratory data analysis |

*Keep the listed technologies aligned with the dependencies and implementations in your repository.*

## Project Structure

```text
Text-classifier/
├── data/
│   ├── raw/                  # Source datasets
│   └── processed/            # Cleaned or transformed data, if saved
├── models/                   # Saved trained models
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Text cleaning and preprocessing
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Single and batch prediction
├── tests/
│   └── test_pipeline.py      # Automated pipeline tests
├── notebooks/
│   └── exploration.ipynb     # Exploratory analysis
├── cli.py                    # Command-line entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

The `processed/` directory and `models/` directory may be populated during execution. Generated datasets and model files do not need to be committed unless they are small, appropriate for distribution, and permitted by their licences.

## Getting Started

### Prerequisites

- Python 3.8 or a compatible version supported by the project's dependencies.
- pip for installing Python packages.
- Git for cloning the repository.
- A labelled news dataset in the format expected by the training script.

### 1. Clone the Repository

```bash
git clone https://github.com/Suhanimeh/Text-classifier.git
cd Text-classifier
```

### 2. Create a Virtual Environment

Create an isolated Python environment for the project:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Resources

If your preprocessing code uses tokenization, stopword removal, or lemmatization, download the corresponding NLTK resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Depending on the NLTK version and the functions used by your implementation, additional resources may be required.

### 5. Prepare the Dataset

Place your labelled news dataset in:

```text
data/raw/news.csv
```

The CSV must contain the text and category columns expected by the training pipeline. Confirm the actual column names in `src/train.py` or `src/preprocess.py` before running the commands below.

**Dataset requirements:**
- Each training record should contain text and its corresponding category.
- Category labels should be consistent.
- Missing or invalid records should be handled during data preparation.
- The dataset should be appropriately licensed for your intended use.

## Usage

The project provides a command-line interface through `cli.py`.

The following commands illustrate the intended workflow. The available commands, options, and argument names must match the actual implementation.

### 1. Train a Model

Train a Logistic Regression classifier using the labelled dataset:

```bash
python cli.py train --data data/raw/news.csv --model-out models/classifier.pkl
```

To select a different algorithm:

```bash
python cli.py train --data data/raw/news.csv --model-type svm --model-out models/svm.pkl
```

The training process should preprocess the input text, extract numerical features, fit the selected classifier, and save the trained model.

### 2. Evaluate the Model

Evaluate the saved classifier:

```bash
python cli.py evaluate --data data/raw/news.csv --model models/classifier.pkl
```

For a meaningful estimate of generalisation performance, evaluation should use held-out test data that was not used to train or select the model. If the CLI evaluates on the same dataset used for training, its results should not be presented as independent test performance.

### 3. Predict a Single Text

Classify a news headline:

```bash
python cli.py predict --text "Scientists discover new exoplanet using AI" --model models/classifier.pkl
```

The model should return the predicted category based on the patterns learned during training.

### 4. Predict from a File

Run batch predictions on a text file:

```bash
python cli.py predict --file data/raw/sample_texts.txt --model models/classifier.pkl
```

The file format must match what `src/predict.py` expects. For example, a line-separated text file may contain one headline or article per line.

## Models

The project is designed to support several classification algorithms.

| Model | Description |
|---|---|
| Logistic Regression | A strong, computationally efficient baseline for many text-classification tasks |
| Multinomial Naive Bayes | A probabilistic classifier commonly used with bag-of-words and TF-IDF features |
| Support Vector Machine (SVM) | A classifier that can perform well on high-dimensional text features |
| Random Forest | An ensemble of decision trees that can model nonlinear relationships between numerical features |

The suitability of each algorithm depends on the dataset, text representation, preprocessing choices, and evaluation methodology.

**Model selection:** Compare the models using the same data split and consistent evaluation metrics. Do not assume that a particular algorithm is the best performer without testing it.

## Evaluation Metrics

A text-classification system should be evaluated using multiple metrics rather than accuracy alone.

| Metric | Description |
|---|---|
| Accuracy | Percentage of predictions that are correct |
| Precision | Proportion of predictions for a category that are correct |
| Recall | Proportion of actual examples in a category that are identified |
| F1-score | Harmonic mean of precision and recall |
| Support | Number of actual test examples belonging to each category |
| Confusion Matrix | Shows which categories are correctly classified and which are confused with others |

Precision, recall, and F1-score help reveal category-specific performance, particularly when the dataset contains uneven numbers of examples per category.

### Example Evaluation Output

The following output is illustrative only. Replace it with results generated by your own evaluation pipeline before publishing.

```text
Classification Report:

              precision    recall  f1-score   support

    politics       0.91      0.89      0.90       120
       sport       0.95      0.96      0.95       130
        tech       0.88      0.90      0.89       115
entertainment       0.92      0.91      0.91       105

    accuracy                           0.92       470
```

Actual performance will depend on the dataset, train-test split, text features, preprocessing, and model configuration.

## Testing

Run the project's automated tests:

```bash
pytest tests/ -v
```

The tests should be used to verify expected behaviour across preprocessing, training, evaluation, and prediction components, depending on what `test_pipeline.py` covers.

Passing automated tests demonstrates that the tested functionality behaves as expected; it does not, by itself, establish classification quality on unseen data.

## Limitations

- Classification quality depends on the size, quality, and representativeness of the training dataset.
- News categories can overlap, making some headlines difficult to classify.
- Text preprocessing may remove useful contextual information if applied too aggressively.
- Models can learn dataset-specific patterns that do not generalise to new sources.
- Performance may vary across topics, writing styles, and publication sources.
- A model trained on news data should not automatically be assumed to work well on other types of text.

## Future Improvements

Potential extensions include:

- Compare TF-IDF with alternative text representations.
- Evaluate models using stratified train-test splits and cross-validation.
- Add a confusion-matrix visualisation.
- Introduce configurable preprocessing options.
- Save evaluation reports and prediction results to CSV.
- Add confidence or probability outputs where supported by the selected classifier.
- Test generalisation on a separate dataset from a different news source.
- Package the pipeline as a REST API or a simple web application.

## Author

**Suhani Mehta**

- **GitHub:** [Suhanimeh](https://github.com/Suhanimeh)
