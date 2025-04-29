import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from training_setting import process_data

def save_tokenizer():
    """
    Extract and save the tokenizer from the process_data function
    """
    print("Extracting tokenizer from process_data...")
    
    # Get the tokenizer from process_data
    _, _, _, _, tokenizer, _ = process_data()
    
    # Save the tokenizer to disk
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Tokenizer saved successfully to tokenizer.pickle")

if __name__ == "__main__":
    save_tokenizer()