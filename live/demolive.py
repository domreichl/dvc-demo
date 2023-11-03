import yaml, json
import numpy as np
import tensorflow as tf
from dvclive import Live
from dvclive.keras import DVCLiveCallback


# load & preprocess data
file_names = ["train_labels", "train_images", "test_labels", "test_images"]
data = {fn: np.load("../data/"+fn+".npy") for fn in file_names}
train_images = data["train_images"] / 255.0
test_images = data["test_images"] / 255.0

# build model
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

# train & evaluate model
with Live() as live:
    model.fit(
        train_images,
        data["train_labels"],
        epochs=yaml.safe_load(open("params.yaml", "r"))["epochs"],
           callbacks=[DVCLiveCallback(live=live)]
    )
    model.save("demomodel")
    live.log_artifact("demomodel", type="model")
    test_loss, test_acc = model.evaluate(test_images,  data["test_labels"], verbose=2)
    live.log_metric("test/loss", test_loss)
    live.log_metric("test/acc", test_acc)
