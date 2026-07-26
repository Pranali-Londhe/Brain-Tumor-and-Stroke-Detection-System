# NeuroDetect AI

An interactive, deep-learning based **brain tumor (MRI)** and **stroke (CT)** detection
system: two CNNs (a from-scratch architecture and an EfficientNetB0 transfer-learning
option), a Grad-CAM explainability layer, and a polished web UI where you drag/drop a
scan and watch it get analyzed with a live scanning animation, confidence bars, and a
heatmap overlay of what the model focused on.

> ⚠️ **This is a research / educational tool, not a medical device.** It is not
> validated for clinical use and must never be used to make real diagnostic or
> treatment decisions. Predictions should only ever be interpreted by qualified
> medical professionals. See "Responsible use" at the bottom.

## What's inside

```
neurodetect-ai/
├── app.py                    # Flask backend + inference API
├── models/
│   ├── tumor_cnn.py           # 4-class tumor CNN (glioma/meningioma/pituitary/no_tumor)
│   ├── stroke_cnn.py          # 3-class stroke CNN (hemorrhagic/ischemic/normal)
│   └── gradcam.py             # Grad-CAM heatmap generation
├── train/
│   ├── dataset_utils.py       # Data loading + light augmentation
│   ├── train_tumor.py         # Training script for the tumor model
│   └── train_stroke.py        # Training script for the stroke model
├── templates/index.html        # UI markup
├── static/css/style.css        # "Scanner room" visual design
├── static/js/main.js            # Upload, scan animation, results rendering
├── saved_models/                # Trained .keras weights go here
└── requirements.txt
```

## Quickstart (UI demo, no dataset needed)

The app works immediately even without a trained model — it runs in **demo mode**:
the real CNN architecture is used end-to-end (upload → inference → Grad-CAM), but
since the network hasn't seen any training data, its predictions are meaningless.
This is clearly labeled in the UI (a red banner + `demo_mode: true` in the API
response) so it's never mistaken for a real result. It's there so you can see the
full interactive experience immediately.

```bash
cd neurodetect-ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## Training on real data

To get real predictions, train each model on a public dataset and drop the
resulting `.keras` file into `saved_models/`.

### Brain tumor MRI
Recommended dataset: **Brain Tumor MRI Dataset** (Kaggle, user `masoudnickparvar`),
~7,000 images across `glioma`, `meningioma`, `pituitary`, `notumor`.

```bash
# after downloading & unzipping into e.g. ./data/tumor (train/ and test/ subfolders,
# one folder per class)
cd train
python train_tumor.py --data_dir ../data/tumor --arch transfer --epochs 25 \
    --out ../saved_models/tumor_model.keras
```

### Brain stroke CT
Recommended dataset: a CT hemorrhage/ischemia dataset such as **"Brain Stroke CT
Image Dataset"** (Kaggle) or the **RSNA Intracranial Hemorrhage Detection** dataset
(you'll need to bucket subtypes like epidural/subdural/subarachnoid into a single
`hemorrhagic` folder, and combine any infarct-labeled scans into `ischemic`).

```bash
cd train
python train_stroke.py --data_dir ../data/stroke --arch transfer --epochs 25 \
    --out ../saved_models/stroke_model.keras
```

Both scripts support `--arch custom` for a lighter from-scratch CNN if you don't
want to fine-tune EfficientNetB0 (faster to train, typically a few points lower
accuracy).

Once a `tumor_model.keras` / `stroke_model.keras` file exists in `saved_models/`,
restart `app.py` — it auto-detects trained weights and switches that scan mode out
of demo mode automatically (see the status pill in the top-right of the UI).

## How inference + Grad-CAM works

1. The uploaded image is resized to 224×224 and passed through the CNN.
2. `models/gradcam.py` hooks into the last convolutional feature map (named
   `gradcam_target_layer` in both architectures) and computes the gradient of the
   predicted class with respect to those features — the classic Grad-CAM
   algorithm — producing a heatmap of "where the model looked."
3. The heatmap is color-mapped and blended over the original slice and sent back
   to the browser as a toggleable overlay, so predictions are explainable rather
   than a black-box percentage.

## Responsible use

- Do not use this system, or any version fine-tuned from it, to diagnose real
  patients or to inform real treatment decisions.
- Model performance depends entirely on the training data — dataset size, imaging
  protocol, scanner vendor, and demographic coverage all affect generalization.
  Always report accuracy, sensitivity/specificity, and confusion matrices per
  class, and validate on data from more than one source before drawing conclusions.
- Class imbalance is common in medical imaging datasets; keep an eye on recall for
  the minority (disease-positive) classes specifically, not just overall accuracy.
- Grad-CAM shows correlation, not causation, in the model's attention. It helps
  catch obviously wrong shortcuts (e.g. focusing on scan borders or text overlays)
  but is not a certification of correctness.
