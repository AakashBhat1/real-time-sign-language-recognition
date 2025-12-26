from function import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
import numpy as np
import os

# Label encoding
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# Load data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        valid = True
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if not os.path.exists(file_path):
                print(f"Missing: {file_path}")
                valid = False
                break
            res = np.load(file_path, allow_pickle=True)
            window.append(res)
        if valid:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
if len(labels) == 0:
    print("Error: No valid sequences found. Please check if your `.npy` files exist.")
    exit()

y = to_categorical(labels).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard logs
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model with validation split
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_split=0.1)

# Model summary
model.summary()

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Final accuracy values
train_accuracy = history.history['categorical_accuracy'][-1]
val_accuracy = history.history['val_categorical_accuracy'][-1]
print(f"\n✅ Final Training Accuracy: {train_accuracy:.4f}")
print(f"✅ Final Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set
loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n📊 Test Accuracy: {test_accuracy:.4f}")
print(f"📉 Test Loss: {loss:.4f}")
