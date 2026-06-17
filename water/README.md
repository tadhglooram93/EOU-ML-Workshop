# 💧 Can AI Predict Irrigation?

A beginner-friendly machine learning activity for the Farm + AI workshop.
Students use recent weather data to train a small model that predicts how much
irrigation a field needed. **No coding or AI background needed.**

This is an educational demo only. It is not meant to make real irrigation
decisions.

> 🔗 **Launch in Binder:** 
https://hub.gesis.mybinder.org/user/tadhglooram93-eou-ml-workshop-wci25ink/doc/workspaces/auto-5/tree/water/irrigation_model.ipynb

## The question we explore

> Can we use recent weather to predict how much irrigation happened on a farm?

Along the way, students discover one of the most important ideas in machine
learning:

> **Good clues (features) matter more than a fancy model.**

## The good-clues vs junk-clues lesson

The notebook gives students two kinds of "clues" to choose from:

- **Good clues** — real weather measurements that actually relate to irrigation.
- **Junk clues** — silly made-up numbers that have nothing to do with water.

The first models are trained **on purpose** with only the junk clues, so they do
a bad job. Students then swap in the good clues and watch the score jump. The
takeaway: even a powerful model cannot fix bad data.

### Good clues (used by the model)

| Variable            | What it measures                              | Why it matters                          |
| ------------------- | --------------------------------------------- | --------------------------------------- |
| `max_temp_f`        | Highest temperature that day                  | Hotter days usually need more water     |
| `mean_humidity_pct` | Average humidity                              | Dry air increases water loss            |
| `avg_wind_mph`      | Average wind speed                            | Wind dries out the field                |
| `precip_in`         | Rain that day                                | Rain means less irrigation is needed    |
| `temp_3day_avg_f`   | Average temperature over the last 3 days       | A "moving average" that captures recent heat |

### Junk clues (added on purpose, totally useless)

| Variable             | What it measures                  |
| -------------------- | --------------------------------- |
| `lucky_number`       | The farmer's lucky number         |
| `barn_cat_naps`      | How many naps the barn cat took   |
| `dice_roll`          | A roll of a six-sided dice        |
| `tractor_color_code` | A code for the tractor's color    |
| `songs_on_radio`     | Songs played on the tractor radio |

## What students do

1. Explore the data and plot weather against irrigation.
2. Use a dropdown to see which clues show a real pattern and which look random.
3. Train **Linear Regression** and a **small Neural Network** — first with junk
   clues (bad results), then with good clues (better results).
4. Read the score: **Mean Absolute Error (MAE)** and **R²**.
5. Enter today's weather and ask the model to predict irrigation.

## The prediction target

```text
irrigation_amount
```

This is a practice target: a simple estimate of how much water the field needed
that day (roughly water lost to the weather minus the rain that fell). It is not
real farmer irrigation, just a good way to test the models.

## How to run

The notebook is designed for **Binder** (use the link above) so students don't
have to install anything. To run it locally instead:

```bash
cd water
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook irrigation_model.ipynb
```

## Folder structure

```text
water/
├── irrigation_model.ipynb        # the student notebook
├── requirements.txt
├── data/
│   ├── model_dataset.csv         # one row per day: weather + irrigation_amount
│   ├── model_metadata.json       # target and column info
│   └── weather.csv               # raw daily weather
└── scripts/
    ├── weather.py                # fetch daily weather for a location
    ├── build_model_dataset.py    # build the model-ready dataset
    └── find_best_features_pycaret.py  # (instructor) feature exploration
```

> Tip: find the latitude/longitude for a field at
> [gps-coordinates.net](https://www.gps-coordinates.net/) when collecting weather.

## Big ideas

By the end, students should understand:

- Machine learning learns patterns from examples.
- The quality of the data (the clues you pick) matters a lot.
- Weather affects irrigation, but it is not the whole story.
- Models can be useful and still be imperfect.

## Important note

This project is for learning only. Actual irrigation decisions should be made by
farmers, agronomists, irrigation specialists, and appropriate farm management
systems.
