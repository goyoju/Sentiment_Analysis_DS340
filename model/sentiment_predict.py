import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import time
from dotenv import load_dotenv
from reddit_api import fetch_reddit_comments

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
is_tf_2_11_plus = int(tf.__version__.split('.')[0]) > 2 or (int(tf.__version__.split('.')[0]) == 2 and int(tf.__version__.split('.')[1]) >= 11)
model_extension = '.keras' if is_tf_2_11_plus else '.h5'

def sentiment_predict(texts, tokenizer, model, max_len=500):
    """
    Predict sentiment of multiple texts
    """
    # Preprocess texts
    processed_texts = []
    for text in texts:
        # preprocessing in the same way
        processed = re.sub(r'[^a-zA-Z ]', '', text).lower()
        processed_texts.append(processed)
    
    # Tokenize and padding
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    
    # Make predictions
    predictions = model.predict(padded, verbose=0)
    
    # Extract scores and labels
    results = []
    for i, pred in enumerate(predictions):
        score = float(pred[0])
        #if the score larger than 0.5, positive, otherwise, negative
        label = "POSITIVE" if score > 0.5 else "NEGATIVE"
        results.append({
            'text': texts[i],
            'score': score,
            'label': label
        })
    
    return results

def load_tokenizer():
    """Load tokenizer from available files"""
    possible_pickle_paths = ['tokenizer.pickle', os.path.join('..', 'tokenizer.pickle')]
    possible_json_paths = ['tokenizer.json', os.path.join('..', 'tokenizer.json')]
    
    # Try pickle files
    for path in possible_pickle_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as handle:
                    return pickle.load(handle)
            except Exception as e:
                print(f"Error loading pickle file: {e}")
    
    # Try JSON files
    for path in possible_json_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    json_string = f.read()
                    return tokenizer_from_json(json_string)
            except Exception as e:
                print(f"An error occurred: {e}")

    # Create new tokenizer if all attempts fail
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(["sample text"])
    return tokenizer

def load_sentiment_model():
    """Load the sentiment analysis model"""
    try:
        possible_paths = [
            f"best_model{model_extension}",
            f"best_model{'.h5' if model_extension == '.keras' else '.keras'}",
            os.path.join("..", f"best_model{model_extension}"),
            os.path.join("..", f"best_model{'.h5' if model_extension == '.keras' else '.keras'}")
        ]
        
        # finding the model
        for path in possible_paths:
            if os.path.exists(path):
                model = load_model(path, compile=False)
                return model
        
        # if the model not found
        raise FileNotFoundError("Model file not found")
    except Exception as e:
        print(f"Model loading error: {e}")
        raise

def analyze_keyword(keyword, limit=100):
    """
    Analyze Reddit comments for a specific keyword
    """
    print(f"Analyzing sentiment for keyword: '{keyword}'...")

    # Fetch comments from Reddit using the imported function
    comments = fetch_reddit_comments(keyword, limit=limit)
    
    if not comments or len(comments) < 1:
        return {
            'error': 'No comments found for this keyword',
            'keyword': keyword
        }
    
    # Load model and tokenizer
    tokenizer = load_tokenizer()
    model = load_sentiment_model()
    
    print(f"Analyzing sentiment of comments..")
    
    # Analyze sentiment of comments
    results = sentiment_predict(comments, tokenizer, model)
    
    # Calculate average sentiment and positive ratio
    avg_sentiment = sum(result['score'] for result in results) / len(results)
    positive_ratio = sum(1 for result in results if result['score'] > 0.5) / len(results)
    
    # Prepare sample comments with sentiment scores
    sample_comments = results[:5] if len(results) > 5 else results
    
    # Return complete result object
    return {
        'keyword': keyword,
        'average_sentiment': avg_sentiment,
        'positive_ratio': positive_ratio,
        'total_comments': len(comments),
        'sample_comments': sample_comments,
        'timestamp': time.time()
    }

def get_sentiment_text(score):
    """Return sentiment description based on score"""
    if score >= 0.75:
        return "Very Positive"
    elif score >= 0.6:
        return "Positive"
    elif score >= 0.4:
        return "Neutral"
    elif score >= 0.25:
        return "Negative"
    else:
        return "Very Negative"

def display_results(results):
    """
    Display analysis results in a nice format in the terminal
    """
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    sentiment_text = get_sentiment_text(results['average_sentiment'])
    
    # Print header
    print("\n" + "="*60)
    print(f"REDDIT SENTIMENT ANALYSIS RESULTS FOR '{results['keyword']}'")
    print("="*60)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"Total comments analyzed: {results['total_comments']}")
    print(f"Average sentiment score: {results['average_sentiment']*100:.1f}%")
    print(f"Sentiment classification: {sentiment_text}")
    print(f"Positive comments ratio: {results['positive_ratio']*100:.1f}%")
    
    # Print interpretation
    print("\nINTERPRETATION:")
    print(f"Based on {results['total_comments']} Reddit comments containing '{results['keyword']}',")
    print(f"the overall sentiment is {sentiment_text.lower()} with an average score of {results['average_sentiment']*100:.1f}%.")
    
    if results['positive_ratio'] > 0.5:
        print(f"The majority ({results['positive_ratio']*100:.1f}%) of comments are positive.")
    else:
        print(f"Only {results['positive_ratio']*100:.1f}% of comments are positive.")
    
    # Print sample comments
    print("\nSAMPLE COMMENTS:")
    for i, comment in enumerate(results['sample_comments']):
        sentiment_text = get_sentiment_text(comment['score'])
        trunc_text = comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text']
        print(f"\n{i+1}. [{comment['score']*100:.0f}% - {sentiment_text}] {trunc_text}")

def single_sentence_analysis():
    """Analyze sentiment of a single input sentence"""
    
    # Load model and tokenizer
    tokenizer = load_tokenizer()
    model = load_sentiment_model()
    
    while True:
        # Get input text
        text = input("\nEnter text to analyze (or 'back' to return): ")
        
        if text.lower() == 'back':
            break
        
        if not text.strip():
            print("Please enter valid text.")
            continue
        
        # Analyze sentiment
        result = sentiment_predict([text], tokenizer, model)[0]
        sentiment_text = get_sentiment_text(result['score'])
        
        # Display result
        print("\nRESULT:")
        print(f"Sentiment score: {result['score']*100:.1f}%")
        print(f"Classification: {sentiment_text}")
        print(f"Confidence: {max(result['score'], 1-result['score'])*100:.1f}%")

def main():
    """Main function to run the Reddit sentiment analyzer from command line"""
    # Load environment variables
    load_dotenv()
    
    print("Analyze sentiment of Reddit comments for a given keyword\n")
    
    # Check if model and tokenizer can be loaded
    try:
        load_tokenizer()
        load_sentiment_model()
    except Exception as e:
        print(f"✗ Failed to load model or tokenizer: {e}")
        return
    
    while True:
        # Display menu
        print("1. Analyze Reddit comments for a keyword")
        print("2. Analyze sentiment of a single sentence")
        print("3. Exit")
        
        # Get user choice
        choice = input("\nEnter your number (1-3): ")
        
        if choice == '1':
            # Get keyword from user
            keyword = input("\nEnter a keyword: ")
            
            if not keyword.strip():
                print("Please enter a valid keyword.")
                continue
            
            # Get comment limit from user
            limit_input = input("Enter maximum number of comments to analyze: ")
            try:
                limit = int(limit_input) if limit_input.strip() else 100
                if limit <= 0:
                    raise ValueError("Limit must be positive")
            except ValueError:
                print("Using default limit of 100 comments.")
                limit = 100
            
            # Set starting time for performance measurement
            start_time = time.time()
            
            try:
                # Analyze the keyword
                results = analyze_keyword(keyword, limit=limit)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display the results
                display_results(results)
                
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
        elif choice == '2':
            # Single sentence analysis
            single_sentence_analysis()
            
        elif choice == '3':
            # Exit the program
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()