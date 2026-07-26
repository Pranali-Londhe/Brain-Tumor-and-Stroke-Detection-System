"""
Dataset loading utilities.

Expects data organized in the standard Keras "flow_from_directory" layout:

    data/
      train/
        <class_1>/*.jpg
        <class_2>/*.jpg
        ...
      val/
        <class_1>/*.jpg
        ...
      test/
        <class_1>/*.jpg
        ...

For brain tumor MRI, recommended public dataset (download separately, this
project does not ship any medical images):
    Kaggle: "Brain Tumor MRI Dataset" (masoudnickparvar)
    Classes: glioma, meningioma, pituitary, notumor

For stroke CT, recommended public dataset:
    Kaggle: "Brain Stroke CT Image Dataset" / RSNA Intracranial Hemorrhage Detection
    Classes: hemorrhagic, ischemic, normal (you may need to merge/relabel
    subtype folders such as epidural/subdural/subarachnoid into "hemorrhagic")
"""

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir, img_size=(224, 224), batch_size=32, val_split=0.15, seed=42):
    """
    If data_dir contains train/ val/ (test/) subfolders, load each directly.
    Otherwise, treat data_dir as a single folder of class subfolders and split it.
    """
    import os

    has_explicit_split = os.path.isdir(os.path.join(data_dir, "train"))

    if has_explicit_split:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(data_dir, "train"), image_size=img_size, batch_size=batch_size, label_mode="categorical"
        )
        val_dir = os.path.join(data_dir, "val")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
        )
        test_dir = os.path.join(data_dir, "test")
        test_ds = None
        if os.path.isdir(test_dir):
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical", shuffle=False
            )
        class_names = train_ds.class_names
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=val_split, subset="training", seed=seed,
            image_size=img_size, batch_size=batch_size, label_mode="categorical"
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=val_split, subset="validation", seed=seed,
            image_size=img_size, batch_size=batch_size, label_mode="categorical"
        )
        test_ds = None
        class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_augmentation():
    """Light, medically-sensible augmentation (no vertical flips - brain
    anatomy has a meaningful up/down orientation; small rotations/zooms only)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")
