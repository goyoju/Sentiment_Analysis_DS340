import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json

def parse_example(example_proto):
    """
    Parse TFRecord example
    """
    feature_description = {
        'reviewText': tf.io.FixedLenFeature([], tf.string),
        'positivity': tf.io.FixedLenFeature([], tf.int64)
    }
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    text = example['reviewText']
    label = example['positivity']
    
    return text, label

def load_fasttext_embeddings(path, word_index, vocab_size, embedding_dim):
    """
    Load pre-trained FastText embeddings
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                
                if word in word_index and word_index[word] < vocab_size:
                    vectors = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[word_index[word]] = vectors
    except Exception as e:
        pass
    
    return embedding_matrix

def process_data(embedding_type='fasttext', fasttext_path=None, max_words=10000, max_len=500):
    """
    Process data for model training
    """
    try:
        if os.path.exists('reviews.tfrecord'):
            raw_dataset = tf.data.TFRecordDataset('reviews.tfrecord')
            
            feature_description = {
                'reviewText': tf.io.FixedLenFeature([], tf.string),
                'positivity': tf.io.FixedLenFeature([], tf.int64)
            }
            
            texts = []
            labels = []
            
            for raw_record in raw_dataset:
                example = tf.io.parse_single_example(raw_record, feature_description)
                text = example['reviewText'].numpy().decode('utf-8')
                label = example['positivity'].numpy()
                texts.append(text)
                labels.append(label)
        else:
            texts = [
                "This product is excellent!",
                "I love this item, it's amazing.",
                "This is terrible, do not buy.",
                "Great value for money, highly recommend.",
                "Poor quality, disappointed with purchase."
            ]
            labels = [1, 1, 0, 1, 0]
    
    except Exception as e:
        texts = [
            "This product is excellent!",
            "I love this item, it's amazing.",
            "This is terrible, do not buy.",
            "Great value for money, highly recommend.",
            "Poor quality, disappointed with purchase."
        ]
        labels = [1, 1, 0, 1, 0]
    
    # Tokenize the texts
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    data = pad_sequences(sequences, maxlen=max_len)
    
    # Convert labels to numpy array
    labels = np.array(labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Save tokenizer as JSON
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    
    # Initialize embedding matrix
    embedding_matrix = None
    
    # Create embedding matrix if using FastText
    if embedding_type == 'fasttext' and fasttext_path and os.path.exists(fasttext_path):
        vocab_size = min(max_words, len(tokenizer.word_index) + 1)
        embedding_matrix = load_fasttext_embeddings(fasttext_path, tokenizer.word_index, vocab_size, 300)
    
    return X_train, X_test, y_train, y_test, tokenizer, embedding_matrix

if __name__ == "__main__":   
    # if log directory exist, remove
    if os.path.exists('logs'):
        import shutil
        shutil.rmtree('logs')

    # create new log directory
    log_dir = "logs/fit/"
    os.makedirs('logs/fit', exist_ok=True)

    # Configure TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # setting up gpu for training
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass

    # Process data with FastText embeddings
    X_train, X_test, y_train, y_test, tokenizer, embedding_matrix = process_data(
        embedding_type='fasttext',
        fasttext_path='crawl-300d-2M-subword.vec'  #using fast text word vector
    )
    
    # Save processed data information
    data_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'vocab_size': len(tokenizer.word_index) + 1,
        'max_len': X_train.shape[1],
        'embedding_dim': 300 if embedding_matrix is not None else 250,
    }
    
    with open('training_info.json', 'w') as f:
        json.dump(data_info, f)