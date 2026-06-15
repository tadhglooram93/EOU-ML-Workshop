

## Find lat lon of area we want to collect weather data
https://www.gps-coordinates.net/


# Farm Irrigation Machine Learning Workshop

This repo contains a simple machine learning activity for students learning how data can help explain real agricultural decisions.

Students will visit a farm, learn about center pivot irrigation, collect a field temperature measurement, and then use weather + irrigation data to train a small model that predicts irrigation amount.

This is an educational demo only. It is not meant to make real irrigation decisions.

## Project Goal

We want to answer a simple question:

> Can we use recent weather data to predict how much irrigation happened on a farm?

The model will learn from historical data like:

* Temperature
* Humidity
* Wind
* Rain
* Past irrigation
* Actual irrigation amount from the farmer

Then we will use the weather from the day of the farm visit to predict that day's irrigation and compare it to what actually happened.

## Repo Structure

```text
farm-irrigation-ml/
├── README.md
├── requirements.txt
└── utils/
|   ├── weather.py
├── irrigation_model.ipynb
└── data/
    ├── weather.csv
    ├── irrigation.csv
    └── merged_irrigation_weather.csv
```

## Dataset

The main dataset is one row per day.

Each row represents:

> Weather conditions for a day + how much irrigation happened that day.

Example:

| date       | max_temp_f | mean_humidity_pct | precip_in | avg_wind_mph | irrigation_amount |
| ---------- | ---------: | ----------------: | --------: | -----------: | ----------------: |
| 2024-06-01 |       84.2 |              41.5 |      0.00 |          7.4 |              0.25 |
| 2024-06-02 |       88.1 |              36.2 |      0.00 |          9.1 |              0.30 |
| 2024-06-03 |       76.5 |              58.7 |      0.12 |          5.8 |              0.00 |

## Raw Variables

These are the original variables we may collect from weather data or the farmer.

| Variable              | What it measures                                                               | Why it matters                                                         |
| --------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `date`                | The calendar date for the row                                                  | Lets us match weather and irrigation for the same day                  |
| `max_temp_f`          | Highest air temperature that day, in Fahrenheit                                | Hotter days usually increase crop water demand                         |
| `min_temp_f`          | Lowest air temperature that day, in Fahrenheit                                 | Helps describe daily temperature range                                 |
| `mean_temp_f`         | Average air temperature that day, in Fahrenheit                                | A general measure of how warm the day was                              |
| `student_temp_f`      | Temperature collected by students during the farm visit                        | Lets students connect their field measurement to the model             |
| `mean_humidity_pct`   | Average relative humidity for the day, as a percent                            | Dry air can increase water loss from soil and plants                   |
| `precip_in`           | Total precipitation for the day, in inches                                     | Rain can reduce or delay the need for irrigation                       |
| `avg_wind_mph`        | Average wind speed for the day, in miles per hour                              | Wind can increase evaporation and affect irrigation efficiency         |
| `max_wind_mph`        | Highest wind speed or gust for the day                                         | Very windy days may affect irrigation decisions                        |
| `et0_in`              | Reference evapotranspiration, in inches                                        | Estimate of how much water could be lost from a reference crop surface |
| `irrigation_amount`   | Amount of irrigation applied that day                                          | This is the value the model tries to predict                           |
| `irrigation_unit`     | Unit for irrigation amount, such as inches, hours, gallons, or percent setting | Helps us interpret the target correctly                                |
| `pivot_runtime_hours` | Number of hours the pivot ran, if available                                    | Another way to measure irrigation activity                             |
| `pivot_speed_pct`     | Pivot speed or percent setting, if available                                   | A setting that may control how much water is applied                   |
| `field_name`          | Field or crop name, if there are multiple fields                               | Different fields or crops may need different water amounts             |
| `notes`               | Farmer notes about unusual days                                                | Helps identify days that may confuse the model                         |

## Engineered Variables

Engineered variables are new columns we create from the raw data.

They help the model understand patterns over time.

| Variable                     | What it measures                                                            | Why it matters                                                |
| ---------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `temp_3day_avg_f`            | Average temperature over the current day and previous 2 days                | Farmers may respond to several hot days, not just one hot day |
| `temp_5day_avg_f`            | Average temperature over the current day and previous 4 days                | Captures longer heat trends                                   |
| `precip_3day_sum_in`         | Total rain over the current day and previous 2 days                         | Recent rain may reduce irrigation need                        |
| `precip_5day_sum_in`         | Total rain over the current day and previous 4 days                         | Captures longer wet periods                                   |
| `wind_3day_avg_mph`          | Average wind speed over the current day and previous 2 days                 | Windy stretches may increase water loss                       |
| `et0_3day_sum_in`            | Total reference evapotranspiration over the current day and previous 2 days | Estimates recent atmospheric water demand                     |
| `days_since_rain`            | Number of days since measurable rainfall                                    | Longer dry periods may increase irrigation need               |
| `previous_irrigation_amount` | Irrigation amount from the previous day                                     | Farmers may avoid irrigating the same way every day           |
| `days_since_irrigation`      | Number of days since the field was last irrigated                           | Helps model irrigation timing                                 |
| `day_of_year`                | Day number within the year, from 1 to 365                                   | Captures seasonal changes                                     |
| `month`                      | Month number, from 1 to 12                                                  | Simpler version of seasonality                                |

## Prediction Target

The main prediction target is:

```text
irrigation_amount
```

This is what the model is trying to predict.

Depending on the farmer's data, this could mean:

* Inches of water applied
* Hours the pivot ran
* Gallons of water used
* Pivot speed or percent setting
* Whether irrigation happened at all

If we only have yes/no irrigation data, the problem becomes a classification task:

```text
Did irrigation happen today?
Yes or No
```

If we have amount data, the problem becomes a regression task:

```text
How much irrigation happened today?
```

## Models We May Try

We will start simple.

Possible models:

1. Linear Regression
   Learns a simple formula from the data.

2. Random Forest
   Learns more flexible patterns and can handle nonlinear relationships.

3. Small Neural Network
   A simple example of the kind of model students may have heard about in AI.

The goal is not to build the most accurate model. The goal is to understand how data becomes a prediction.

## Student Activity

Students will:

1. Look at the farm and pivot irrigation system.
2. Collect a temperature measurement.
3. Load historical weather and irrigation data.
4. Create simple features like rolling average temperature.
5. Train a model.
6. Test the model on past days.
7. Predict irrigation for the day of the farm visit.
8. Compare the model's prediction to what actually happened.

## Big Ideas

By the end, students should understand:

* Machine learning learns from examples.
* The quality of the data matters.
* Weather affects irrigation decisions, but it is not the whole story.
* Models can be useful and still be imperfect.
* Real-world decisions often include human judgment, equipment limits, and context that may not appear in the data.

## Important Note

This project is for learning only.

Actual irrigation decisions should be made by farmers, agronomists, irrigation specialists, and appropriate farm management systems.
