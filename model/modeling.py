import os
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, Dropout, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from training_setting import process_data
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import json

# checking tensorflow version
tf_version = tf.__version__
is_tf_2_11_plus = int(tf_version.split('.')[0]) > 2 or (int(tf_version.split('.')[0]) == 2 and int(tf_version.split('.')[1]) >= 11)

# Check if we already have processed data from training_setting.py
if os.path.exists('training_info.json'):
    with open('training_info.json', 'r') as f:
        training_info = json.load(f)
    process_again = False
else:
    process_again = True

# if log directory exist, remove
if os.path.exists('logs'):
    import shutil
    shutil.rmtree('logs')

# create new log directory
log_dir = "logs/fit/"
os.makedirs('logs/fit', exist_ok=True)

# Configure TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# setting gpu for training
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    pass

# Process data again if needed
X_train, X_test, y_train, y_test, tokenizer, embedding_matrix = process_data(
    embedding_type='fasttext',
    fasttext_path='crawl-300d-2M-subword.vec'  #using fast text word vector
)

embedding_dim = 300 if embedding_matrix is not None else 250
vocab_size = min(10000, len(tokenizer.word_index) + 1)
max_len = X_train.shape[1]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Calculate class weights to handle imbalance
pos_samples = np.sum(y_train)
neg_samples = len(y_train) - pos_samples
total_samples = len(y_train)

if abs(pos_samples/total_samples - 0.5) > 0.1:
    weight_for_0 = (1 / neg_samples) * total_samples / 2.0
    weight_for_1 = (1 / pos_samples) * total_samples / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
else:
    class_weight = None

# building model
print("Building model...")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Input layer
    inputs = tf.keras.Input(shape=(max_len,))
    
    # Embedding layer
    if embedding_matrix is not None:
        x = Embedding(
            vocab_size,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=False
        )(inputs)
    else:
        x = Embedding(
            vocab_size,
            embedding_dim
        )(inputs)
    
    # using LSTM for neural network training
    x = Bidirectional(LSTM(128, return_sequences=True, 
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(GRU(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)))(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # locate the saving path
    model_path = 'best_model.keras' if is_tf_2_11_plus else 'best_model.h5'
    
    # Callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                 tf.keras.metrics.AUC()]
    )
    
    # train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=15,  
        callbacks=[es, mc, reduce_lr, tensorboard_callback],
        batch_size=32,  
        validation_split=0.2,
        class_weight=class_weight,  
        verbose=2
    )
    
    # saving model
    model.save(model_path)

    # final report
    try:
        model_final = load_model(model_path)
    except:
        print("Error loading saved model. Using current model for evaluation.")
        model_final = model
        
    print("Modeling process completed successfully.")