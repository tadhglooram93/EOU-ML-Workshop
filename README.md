# 🌾 Farm + AI Machine Learning Workshop

A hands-on introduction to machine learning for GO-STEM AI + AG summer camp. **No coding or AI background needed.** Each notebook is
written like a recipe: run the cells from top to bottom, read the short notes,
change a few values, and watch what the model does!

The big questions we explore:

> Can a computer learn patterns from farm data?
> What makes a model good or bad at its job?

---

## The two learning exercises

### 1. 🌱 Crops: Can AI tell crops apart from a photo?

Train an image classifier that looks at a crop photo and predicts whether it is
**alfalfa** or **garbanzo (chickpea)**. Students learn about **transfer
learning**, then upload a photo they took from the farm and let the model classify it.

- Type of ML: **image classification** using transfer learning (MobileNetV2)
- Folder: [`crops/`](crops/) · details in [`crops/README.md`](crops/README.md)

> 🔗 **Launch in Binder:** https://2i2c.mybinder.org/v2/gh/tadhglooram93/EOU-ML-Workshop/HEAD?filepath=%2Fcrops%2Fcrop_classifier_mobilenetv2.ipynb%2FHEAD&urlpath=%2Fdoc%2Ftree%2Fcrop_classifier_mobilenetv2.ipynb


### 2. 💧 Water: Can AI predict irrigation?

Use real weather data (temperature, humidity, wind, rain) to train a model that
predicts how much a field was irrigated. Along the way, students learn why
**good clues (features) matter more than a fancy model** by comparing helpful
weather clues against silly "junk" clues like a lucky number or a dice roll.

- Type of ML: **regression** (predicting a number) with tabular data
- Models: Linear Regression and a small Neural Network
- Folder: [`water/`](water/) · details in [`water/README.md`](water/README.md)

> 🔗 **Launch in Binder:** https://hub.2i2c.mybinder.org/user/tadhglooram93-eou-ml-workshop-3xsh1qbu/doc/tree/water/irrigation_model.ipynb

---

## How students use this

1. Open the Binder link for an exercise (added above).
2. Run each cell from top to bottom.
3. Read the notes and answer the questions in your group.
4. Experiment! Change the inputs and see how the model's results change.

## A note for everyone

These projects are for **learning only**. Real farm decisions should be made by
farmers, agronomists, and specialists. Machine learning can help us ask better
questions, but it does not replace real-world expertise.
