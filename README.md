# Uber Demand Prediction

The project predicts the demand for Uber rides using [**Taxi in New York City**](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2016-03.csv). So, that drivers can be better prepared for the demand, navigate to regions of higher demand and make more money.

We predict number of pickups at a given time interval (e.g., 15 minutes, 30 minutes, or 1 hour such that below 3 conditions are met) in a given region.
- Longer time interval enables drivers to reach the region with high demand
- Longer time interval causes cost and time wastage
- Longer time interval many not be accurate.

In New York, cabs take 15 minutes to travel 1 mile of distance. So, centroids of current region and neighboring regions should be around 1 mile apart, thus time interval should be around 15 minutes. Also, Regions should not be too small that it becomes individual pickup points.

**KMeans Clustering** stores the **centroids** for each regions in the cluster and we will use this centroid to calculate the distances from the near by regions and sort the distances to the near by regions. These centroids server as identity for each region. We will be using `mini-batch KMeans clustering` to break down the city into regions because we have a very large dataset (like here we have crores of rows) and mini-batch KMeans is more efficient than KMeans clustering.

We will choose 8(north, south, east, west, north-east, north-west, south-east, south-west) neighboring regions for each region.

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

1. concat jan, feb and mar data
2. drop unnecessary columns and remove outliers
3. use latitude and longitude from data to scale and cluster using KMeans and save the scaler and KMeans model for later use
4. Resample the data to get total number of pickups in 15 minutes intervals and calculate average pickups for each time interval in each region using EWMA
5. add lag features like t-1, t-2, t-3 and t-4.
6. split the resample data into train and test set on the basis of time like jan and feb data will be used for training and mar data will be used for testing.
7. train the model after encoding the categorical features and evaluate using test data.
8. log the model to mlflow
9. register the model to mlflow model registry
10. change the alias of the model to production and deploy it.

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

