# 🌱 Crop Image Classifier: Alfalfa vs Garbanzo

A low-code machine learning activity where students train an image classifier
that looks at a crop photo and predicts one of two labels:

- **alfalfa**
- **garbanzo** (also called chickpea)

This is an educational demo for the Farm + AI workshop. It is not meant to make
real agricultural decisions.

> 🔗 **Launch in Binder:** https://hub.2i2c.mybinder.org/user/tadhglooram93-eou-ml-workshop-dbd9r2ng/doc/tree/crops/crop_classifier_mobilenetv2.ipynb

## Big idea

A neural network learns from examples. Here, each example is:

> image → crop label

We use **transfer learning**: we start from **MobileNetV2**, a model that has
already learned useful image features from millions of photos, then teach it our
two crop classes. This is much faster than training a model from scratch.

## What students do

1. Load the prepared crop images (already sorted into class folders).
2. Look at a few training images and ask: do the labels look right? Could the
   model "cheat" by looking at the background instead of the plant?
3. Build the model (frozen MobileNetV2 + a small classifier on top).
4. Train it and read the learning curve.
5. Test it on images it has never seen.
6. Take a photo at the farm, upload it, and let the model classify it.

## How to run

The notebook is designed for **Binder** (use the link above) so students don't
have to install anything. To run it locally instead:

```bash
cd crops
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook crop_classifier_mobilenetv2.ipynb
```

If training is too slow, lower `EPOCHS`, `BATCH_SIZE`, or `LEARNING_RATE` in the
settings cell, or skip training and load the saved model in
`models/crop_classifier_mobilenetv2.keras`.

## Folder structure

```text
crops/
├── crop_classifier_mobilenetv2.ipynb   # the student notebook
├── requirements.txt
├── data/
│   ├── raw_crop_images/                 # original downloaded photos (by class)
│   └── crop_images/                     # prepared, resized images
│       ├── train/   alfalfa/  garbanzo/
│       ├── val/     alfalfa/  garbanzo/
│       ├── test/    alfalfa/  garbanzo/
│       └── dataset_manifest.csv
├── models/
│   ├── crop_classifier_mobilenetv2.keras   # a pre-trained model students can load
│   └── class_names.json
├── uploaded_field_photos/               # photos students take at the farm
└── scripts/
    ├── download_images.py               # download photos from an observations CSV
    └── prepare_crop_dataset.py          # crop/resize and split into train/val/test
```

## Preparing the data (instructors)

The images come from public nature-observation data. The two helper scripts let
you rebuild the dataset:

```bash
# 1. Download photos listed in an observations CSV (run from repo root)
python crops/scripts/download_images.py --file <observations.csv> --out <folder>

# 2. Manual Review and filering of photos 
...

# 3. Center-crop, resize, and split into train/val/test folders
python crops/scripts/prepare_crop_dataset.py
```

The resulting folder layout works with both Keras
`image_dataset_from_directory` and Google Teachable Machine.

## Reflection questions

1. Did the model get the field photo right?
2. What part of the plant do you think the model noticed?
3. What could confuse the model (young plants, wet leaves, blurry photos)?
4. How could we improve the training data?

## Important note

This project is for learning only. Real crop identification and farm decisions
should be made by farmers, agronomists, and specialists.
