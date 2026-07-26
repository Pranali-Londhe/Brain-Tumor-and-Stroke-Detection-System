"""
Train the brain stroke CT classifier.

Usage:
    python train_stroke.py --data_dir /path/to/stroke_data --arch transfer --epochs 25
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from models.stroke_cnn import build_custom_cnn, build_transfer_model, compile_model, IMG_SIZE
from train.dataset_utils import load_datasets, build_augmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to stroke CT dataset")
    parser.add_argument("--arch", choices=["custom", "transfer"], default="transfer")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="../saved_models/stroke_model.keras")
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size
    )
    print("Detected classes:", class_names)

    augment = build_augmentation()
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    if args.arch == "custom":
        model = build_custom_cnn(input_shape=IMG_SIZE + (3,), num_classes=len(class_names))
    else:
        model = build_transfer_model(input_shape=IMG_SIZE + (3,), num_classes=len(class_names))

    model = compile_model(model, lr=args.lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if test_ds is not None:
        loss, acc, auc = model.evaluate(test_ds)
        print(f"Test accuracy: {acc:.4f}  |  Test AUC: {auc:.4f}")

    model.save(args.out)
    print(f"Saved model to {args.out}")

    with open(os.path.join(os.path.dirname(args.out), "stroke_class_names.txt"), "w") as f:
        f.write("\n".join(class_names))


if __name__ == "__main__":
    main()
