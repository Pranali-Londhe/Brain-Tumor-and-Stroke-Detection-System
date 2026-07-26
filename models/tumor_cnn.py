"""
Brain Tumor Classification CNN
==============================
Classifies a brain MRI slice into one of four classes:
    0: glioma
    1: meningioma
    2: pituitary
    3: no_tumor

Two architectures are provided:
  - build_custom_cnn(): a from-scratch CNN, good for learning / small compute budgets.
  - build_transfer_model(): EfficientNetB0 transfer-learning model, recommended for
    best accuracy (this is what most published brain-tumor-MRI papers use as a baseline).

Input: 224x224x3 RGB (MRI slices are grayscale but we replicate to 3 channels so
transfer-learning backbones pretrained on ImageNet can be used).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model

IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=NUM_CLASSES) -> Model:
    """A compact, from-scratch CNN. ~2.3M params. Trains fast, good baseline."""
    inputs = layers.Input(shape=input_shape, name="mri_input")

    x = layers.Rescaling(1.0 / 255)(inputs)

    def conv_block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.ReLU(name=f"{name}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.ReLU(name=f"{name}_relu2")(x)
        x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
        return x

    x = conv_block(x, 32, "block1")
    x = conv_block(x, 64, "block2")
    x = conv_block(x, 128, "block3")
    x = conv_block(x, 256, "block4")

    # This is the layer Grad-CAM will hook into for the custom model.
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="gradcam_target_layer")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="tumor_output")(x)

    model = models.Model(inputs, outputs, name="tumor_custom_cnn")
    return model


def build_transfer_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, fine_tune_at=100) -> Model:
    """EfficientNetB0 backbone pretrained on ImageNet + custom classification head."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    # name this so Grad-CAM can find the last conv feature map easily
    x = layers.Activation("linear", name="gradcam_target_layer")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="tumor_output")(x)

    model = models.Model(inputs, outputs, name="tumor_efficientnet")
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
