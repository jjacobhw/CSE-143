# HW 1 Write-Up

**Team 56:** Jacob Wei, Kasra Mokhtari  
**Note about third group member:** On GitLab, it says we have a third group member, Andrei Ionov, whom we tried to reach out to by email, but they never responded.

## How to Run

From the `hw1` folder, install the required packages:

```bash
pip install -r requirements.txt
```

Then run the setup scripts in this order:

```bash
python3 download_and_split_data.py
python3 datapreprocessing.py
```

This will:
- download and split the IMDb dataset into CSV files
- clean the text and save the processed files into `data/cleaned data/`

After that, open `hw1.ipynb` and run the notebook cells from top to bottom.

If NLTK tokenizer data is missing, it will be downloaded automatically the first time it is needed.

## Data Preprocessing

We decided to use a combination of `.py` files for data cleaning, classification utility functions, and tokenization utility functions. For cleaning the data, we removed the `<br />` tags in the text because they do not add anything meaningful, so we did not want our model to treat them as useful features during training.

We also chose to lowercase all text during preprocessing to reduce sparsity and normalize the text. We did not want capitalization that did not carry semantic meaning to affect tokenization. For example, uppercase `H` has a different token value than lowercase `h`. One situation where lowercasing can help is with words like `Hello`, `hello`, and `HELLO`, which all mean the same thing. However, lowercasing can also lose useful distinctions, such as `Apple` the company versus `apple` the fruit. It can also weaken signals like acronyms or all-caps expressions that may show excitement or tone. Even with those tradeoffs, we decided that lowercasing would be more helpful overall for this sentiment classification task.

## Tokenization

We knew we needed to experiment with different tokenization methods, so we created a separate file to store tokenization utility functions that were later used in our notebook. We split the cleaned text into word-level tokens using four methods: standard whitespace splitting, `ToktokTokenizer`, `word_tokenize`, and `WordPunctTokenizer`.

The main difference we noticed qualitatively was punctuation handling. Simple whitespace splitting often kept punctuation attached to words, while `word_tokenize` and `wordpunct` separated things more cleanly. The differences were small overall, but this may help explain why there were slight performance differences between tokenizers.

## Training and Testing Process and Results

Using our utility files, we wrote code in the notebook to train and evaluate all combinations of our classifiers, vectorizers, and tokenizers on the development set. We made a helper function that takes a classifier type, vectorization method, and tokenization method, then trains the model on the training set and evaluates it on the dev set.

Since we had 3 classifiers, 2 vectorization methods, and 4 tokenization methods, this gave us a total of 24 different model combinations to test. We stored the results for each combination and sorted them by development accuracy, while also recording the F1-score for comparison.

The best-performing model on the dev set was the **SVM classifier with TF-IDF vectorization and the `word_tokenize` tokenizer**, which had a dev accuracy of **0.868** and a dev F1-score of about **0.870**. On the test set, it had an accuracy of **0.86** and an F1-score of **0.858**. These results were fairly close to our dev-set performance, which suggests that the model did not overfit much to the dev data and generalized reasonably well.

One pattern we noticed was that TF-IDF performed better overall than CountVectorizer, since the top 8 models in our results all used TF-IDF. We also noticed that SVM generally performed the best, while logistic regression and naive Bayes were slightly lower on average. The differences between tokenizers were smaller, but `word_tokenize` and `wordpunct` tended to do a little better than the others in the top results.

## Error Analysis

After testing our final model on the test set, we displayed 10 reviews that the model classified incorrectly. In all 10 of those examples, the model predicted that the review was negative when it was actually positive. After looking at them more closely, many of the reviews seemed to be overall positive but included a lot of negative words or phrases such as “mediocre,” “horrible,” or “didn't like it” while criticizing specific parts of the movie.

## Group Contributions

### Jacob Wei
- Data cleaning, preprocessing, and splitting
- Data visualization and analysis in the Jupyter notebook
- Classifier utility script (logistic regression, support vector machine, naive Bayes)
- Report

### Kasra Mokhtari
- Tokenization utility file
- Tokenization and error analysis in the Jupyter notebook
- Training and testing functions/implementation in the Jupyter notebook
- Report