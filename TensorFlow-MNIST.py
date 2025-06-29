import tensorflow as tf
import numpy as np

# --- 1. GPU VERIFICATION ---
# This is the most important part: check if TensorFlow can see the GPU.
print("--- GPU Check ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True to avoid TensorFlow allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Success! TensorFlow has detected {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"- GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print("RuntimeError in GPU setup:", e)
else:
    print("Warning: No GPU detected by TensorFlow. The model will run on the CPU.")
print("--------------------\n")


# --- 2. LOAD AND PREPARE DATA ---
# We will only use a small subset of MNIST for a quick test.
print("--- Loading and Preprocessing MNIST Data ---")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Take a smaller subset for faster training
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

# Normalize pixel values from 0-255 to 0-1. This helps the model train better.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension for the CNN. MNIST images are grayscale, so it's 1.
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Data ready.\n")


# --- 3. BUILD THE MODEL ---
# A simple Convolutional Neural Network (CNN) is great for image tasks.
print("--- Building the Model ---")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10) # Output layer has 10 units for 10 digits (0-9)
])

# Compile the model with an optimizer, loss function, and metrics to track.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary() # Print a summary of the model architecture
print("Model built and compiled.\n")


# --- 4. TRAIN THE MODEL ---
# TensorFlow will automatically use the GPU if it's available.
# The log output during training will show device placement.
print("--- Starting Training ---")
model.fit(
    x_train,
    y_train,
    epochs=3, # Run for only 3 passes over the data for a quick test
    validation_data=(x_test, y_test)
)
print("Training finished.\n")


# --- 5. EVALUATE THE FINAL MODEL ---
print("--- Evaluating Final Model ---")
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal test accuracy: {acc:.4f}")
