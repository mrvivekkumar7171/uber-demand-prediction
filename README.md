# Uber Demand Prediction

The project predicts the demand for Uber rides using [**Taxi in New York City**](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2016-03.csv). So, that drivers can be better prepared for the demand, navigate to regions of higher demand and make more money.

We predict number of pickups at a given time interval (e.g., 15 minutes, 30 minutes, or 1 hour such that below 3 conditions are met) in a given region.
- Longer time interval enables drivers to reach the region with high demand
- Longer time interval causes cost and time wastage
- Longer time interval many not be accurate.

**KMeans Clustering** stores the **centroids** for each regions in the cluster and we will use this centroid to calculate the distances from the near by regions and sort the distances to the near by regions. These centroids server as identity for each region.

Darker regions on the map indicate higher demand and lighter regions indicate lower demand for Uber rides.

Steps:
1. Use **Dask** in chunking for data processing and model training to handle the large dataset efficiently
2. Perform **EDA (Exploratory Data Analysis)** and **Feature Selection**
3. Break down the city into regions using **Unsupervised Learning techniques** like **KMeans Clustering**
4. Break down the time axis into intervals
5. For each region, prepare the historical data using **Time Series Analysis**
6. For each region, train the **Regression Models** to predict the demand for Uber rides at a given time
7. Evaluation using metrics like **MAPE (Mean Absolute Percentage Error)** Because it calculates the absolute percentage difference, it actually penalizes errors on smaller actual values more heavily and can be biased
8. Hyperparameter tune the best-performance model using **Optuna**
9. Plot the regional demand on a map or graph to visualize trends.


> Demand is the number of successful Uber ride service requests at a location and time.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

