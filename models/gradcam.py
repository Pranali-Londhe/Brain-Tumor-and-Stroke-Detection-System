"""
Grad-CAM: visual explanations for the CNN's decision.

Produces a heatmap over the input scan showing which regions the network
weighted most heavily for its predicted class - this is what turns the model
from a "black box" into something a clinician/user can sanity-check.

Both tumor_cnn and stroke_cnn name their last conv feature map
'gradcam_target_layer' so this single utility works for either model.
"""

import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="gradcam_target_layer", pred_index=None):
    """
    img_array: preprocessed batch of shape (1, H, W, 3)
    Returns: heatmap as a (h, w) numpy array normalized to [0, 1]
    """
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_bgr_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    original_bgr_img: original image as np.uint8 array (H, W, 3), BGR
    heatmap: (h, w) float array in [0, 1] from make_gradcam_heatmap
    Returns: overlaid image, np.uint8 (H, W, 3) BGR
    """
    h, w = original_bgr_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    overlaid = cv2.addWeighted(colored_heatmap, alpha, original_bgr_img, 1 - alpha, 0)
    return overlaid
