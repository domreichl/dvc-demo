import yaml, json
import numpy as np
import tensorflow as tf


# load & preprocess data
file_names = ["train_labels", "train_images", "test_labels", "test_images"]
data = {fn: np.load("../data/"+fn+".npy") for fn in file_names}
train_images = data["train_images"] / 255.0
test_images = data["test_images"] / 255.0

# build & train model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(
    train_images,
    data["train_labels"],
    epochs=yaml.safe_load(open("params.yaml", "r"))["epochs"]
)

# evaluate model
test_loss, test_acc = model.evaluate(test_images,  data["test_labels"], verbose=2)
performance = {
    "Loss": {
        "train": history.history["loss"][-1],
        "test": test_loss,
    },
    "Accuracy": {
        "train": history.history["accuracy"][-1],
        "test": test_acc,
    },
}
json.dump(performance, open("performance.json", "w"))

