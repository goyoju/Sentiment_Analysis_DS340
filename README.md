# RedditSense_Sentiment_Analysis_DS340

## Abstract
This project implements a sentiment analysis system that analyzes Reddit comments containing specific keywords to determine public sentiment. The system uses a deep learning model based on bidirectional LSTM and GRU neural networks, trained on a diverse dataset of Amazon product reviews. It provides real-time sentiment analysis of Reddit comments, enabling users to gauge public opinion on topics, products, or events of interest.

- **Input** :  Keyword (e.g., "bitcoin")
- **Outputs** :  Sentiment analysis with metrics including average sentiment score, positive ratio, and sample comments with individual sentiment scores.

### Example:
	1. Analyze Reddit comments for a keyword
    2. Analyze sentiment of a single sentence
    3. Exit

    Enter your number (1-3): 1
    Enter a keyword: bitcoin
    Enter maximum number of comments to analyze: 100

    SUMMARY:
    Total comments analyzed: 100
    Average sentiment score: 54.8%
    Sentiment classification: Neutral
    Positive comments ratio: 57.0%

    INTERPRETATION:
    Based on 100 Reddit comments containing 'bitcoin',
    the overall sentiment is neutral with an average score of 54.8%.
    The majority (57.0%) of comments are positive.

    SAMPLE COMMENTS:
    1. [6% - Very Negative] There were sites that gave away whole bitcoins for solving a captcha. Literally 5 seconds of work pe...
    2. [63% - Positive] Short google search brings up his X profile. Poor guy gave up on mining bitcoin(probably shortly aft...
    3. [98% - Very Positive] My college roommate was mining about 10 Bitcoin a week in 2009. We used to complain about loud the f...
    4. [40% - Negative] I still don't understand what "mining" in this sense actually is. Like what is actually *creating* t...
    5. [76% - Very Positive] It's fascinating, I was talking back in the day with 2 other people on a forum I didn't know. One wa...

## Installation
### Note:
    You can skip this section if you only plan to use the pre-trained model for prediction.
    
### Dataset download
Since the datasets are too big, it is required to download the dataset from the resource:

1. Go to https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
2. Go to "Per-category data" column and download:
    * Books
    * Electronics
    * Clothing, Shoes and Jewelry
    * All Beauty
3. Unzip those datasets and put All_Beauty.json, Books.json, Clothing_Shoes_and_Jewelry.json, Electronics.json in the data folder

### Word embeddings download
 1. Go to FastText embeddings from https://fasttext.cc/docs/en/english-vectors.html
 2. Download crawl-300d-2m-subword.zip
 3. Unzip and put crawl-300d-2M-subword.vec in the model folder

## Setting up

### Set up Reddit API credentials in a .env file:

    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=SentimentAnalyzer/1.0

## Run

1. 
Create and activate a conda environment:
###
    conda create -n new
    conda activate new


2. Install dependencies:

###
    cd model
    pip install -r requirement.txt


3.
### 
    Note : If you want to use the prediction with saved model, skip to step 4

Run the setup script to prepare the model:


    cd ..
    python setting_up.py


This script handles:
* Converting data to TFRecord format
* Processing data for training
* Training the model
* Saving the tokenizer

4. 
Run the prediction script to analyze sentiment:

###
    python prediction.py

This will open an interactive CLI with the following options:

###
    1. Analyze Reddit comments for a keyword
    2. Analyze sentiment of a single sentence
    3. Exit

## Model Performance

The sentiment analysis model achieves high accuracy on the validation dataset:

    * Validation Accuracy: 93.11%
    * Validation Precision: 96.37%
    * Validation Recall: 95.59%
    * Validation AUC: 93.14%

These metrics were achieved at epoch 9 of model training. The high precision and recall values indicate that the model is effective at correctly identifying both positive and negative sentiments in text data.

Training metrics:

    * Training Accuracy: 98.17%
    * Training Precision: 99.81%
    * Training Recall: 98.06%
    * Training AUC: 99.55%

## Datasets

The model is trained on a diverse dataset of Amazon product reviews from multiple categories:

    * Beauty products
    * Books
    * Clothing, Shoes, and Jewelry
    * Electronics

## test environment

* Python 3
* TensorFlow 2.10.0
* Flask 2.0.1
* NumPy 1.24.3
* PRAW 7.5.0
* Python-dotenv 0.19.2
* CUDA : 11.8
* cuDNN : 8.7
* Hardware:
    * GPU : RTX 4060TI 6GB
    * Memory : 32 GB