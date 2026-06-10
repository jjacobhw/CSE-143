# Text Spam Detector

This project builds a text spam detector using TF-IDF features and machine learning classification models. We trained and tested our models on two datasets: the SMS Spam Collection for text messages and the Enron spam dataset for emails. The main model uses a Linear Support Vector Machine (Linear SVM), with Naive Bayes and Logistic Regression included as baseline comparisons. The project flow starts by loading and cleaning the datasets, then converting each message into TF-IDF numerical features, training the models, and evaluating their performance using accuracy, precision, recall, F1, F2, confusion matrices, and 5-fold cross-validation.

Most of the training, testing, and analysis is done in `spam_detection_final.ipynb`, while reusable helper functions are stored in the `.py` files inside the `src/` folder. We also added extra NLP experiments comparing word n-grams, character n-grams, and a combined word + character model to see how different text features affect spam detection performance. After training and evaluating the models, we saved the final models and built a Streamlit demo app where a user can paste in an SMS message and get a spam or non-spam prediction. The app also shows the top model signals, which are the words or features that pushed the Linear SVM toward spam or non-spam, making the prediction easier to interpret.

## Datasets

This project uses two datasets. The SMS dataset is stored as `sms_spam_collection.tsv`, which contains text messages labeled as spam or non-spam. The Enron dataset is stored as `enron_spam_data.csv`, which contains email subjects and messages labeled as spam or non-spam.

The project expects the dataset files to be inside the `data/` folder:

```text
data/
├── sms_spam_collection.tsv
└── enron_spam_data.csv
```

For the SMS dataset, the file is tab-separated and does not have a header row. The first column is the label, and the second column is the message. For the Enron dataset, we combine the `Subject` and `Message` columns into one text field and use the `Spam/Ham` column as the label.

## Project Structure

The project is organized so the notebook contains the main explanation and results, while the Python files contain helper functions for loading data, training models, evaluating results, and explaining predictions.

```text
final/
├── data/
├── models/
├── results/
├── src/
├── spam_detection_final.ipynb
├── streamlit_app.py
├── train_and_save_models.py
├── requirements.txt
└── README.md
```

The `data/` folder stores the datasets, the `models/` folder stores the saved trained models, and the `results/` folder stores output files like confusion matrices and metrics tables. The `src/` folder contains the main helper code used by the notebook and app.

## How to Run the Project

First, create and activate a virtual environment from the main project folder.

On Mac or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

After the requirements are installed, open the main notebook:

```bash
jupyter notebook spam_detection_final.ipynb
```

Once the notebook opens, run all cells from top to bottom. The notebook loads the data, trains and tests the models, creates the evaluation tables, saves the result files, and shows the main analysis.

## How to Train and Save the Models

The notebook trains models for testing and evaluation. To train the final SMS and email models and save them into the `models/` folder, run:

```bash
python train_and_save_models.py
```

This creates saved model files that can be used by the Streamlit app.

## How to Run the Streamlit App

After the models are saved, run the app with:

```bash
streamlit run streamlit_app.py
```

The app lets a user paste in an SMS message and get a spam or non-spam prediction. It also shows the top model signals, which are the words or features from the message that pushed the TF-IDF + Linear SVM model toward spam or non-spam.

## Methods Used

The project starts by cleaning the text, which includes lowercasing messages and replacing URLs, email addresses, and numbers with general tokens. Then the cleaned text is converted into TF-IDF features so the models can work with numerical data instead of raw text.

The main model is a Linear SVM using `LinearSVC` with `class_weight="balanced"`. We also tested Naive Bayes and Logistic Regression as baseline models. To make the NLP side stronger, we compared word n-grams, character n-grams, and a combined word + character model. We also added model interpretability by looking at the Linear SVM feature weights to see which words or phrases pushed the model toward spam or non-spam.

## Results

The notebook saves its main results into the `results/` folder. This includes confusion matrices for both datasets, a metrics summary CSV comparing the models, and the n-gram comparison table and chart.

The main evaluation metrics are accuracy, precision, recall, F1, and F2. We included F2 because it weights recall more heavily, which is useful for spam detection since missing spam can be more risky than simply reporting accuracy alone.