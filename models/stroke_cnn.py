"""
Brain Stroke Classification CNN
================================
Classifies a brain CT slice into one of three classes:
    0: hemorrhagic  (bleed - appears hyperdense/bright on CT)
    1: ischemic     (clot/infarct - appears hypodense/dark, subtler on CT)
    2: normal

Architecture mirrors tumor_cnn.py's design (shared philosophy, separate model
because stroke CT scans and tumor MRI scans have very different intensity/texture
statistics and should not share a single feature extractor).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model

IMG_SIZE = (224, 224)
NUM_CLASSES = 3
CLASS_NAMES = ["hemorrhagic", "ischemic", "normal"]


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=NUM_CLASSES) -> Model:
    inputs = layers.Input(shape=input_shape, name="ct_input")
    x = layers.Rescaling(1.0 / 255)(inputs)

    def conv_block(x, filters, name, dropout=0.0):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.ReLU(name=f"{name}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.ReLU(name=f"{name}_relu2")(x)
        x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
        if dropout:
            x = layers.SpatialDropout2D(dropout, name=f"{name}_sdrop")(x)
        return x

    x = conv_block(x, 32, "block1")
    x = conv_block(x, 64, "block2", dropout=0.1)
    x = conv_block(x, 128, "block3", dropout=0.1)
    x = conv_block(x, 256, "block4", dropout=0.2)

    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="gradcam_target_layer")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="stroke_output")(x)

    model = models.Model(inputs, outputs, name="stroke_custom_cnn")
    return model


def build_transfer_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, fine_tune_at=100) -> Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="ct_input")
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Activation("linear", name="gradcam_target_layer")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="stroke_output")(x)

    model = models.Model(inputs, outputs, name="stroke_efficientnet")
    return model


def compile_model(model: Model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    m = build_custom_cnn()
    m.summary()
