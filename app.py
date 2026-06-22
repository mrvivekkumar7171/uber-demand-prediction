# streamlit run app.py
from sklearn.pipeline import Pipeline
from sklearn import set_config
from pathlib import Path
from time import sleep
import streamlit as st
import datetime as dt
import pandas as pd
import dagshub
import mlflow
import joblib


set_config(transform_output="pandas")

mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/uber-demand-prediction.mlflow")
dagshub.init(repo_owner='mrvivekkumar7171', repo_name='uber-demand-prediction', mlflow=True)

# get model name
registered_model_name = 'uber_demand_prediction_model'
stage = "Staging"
model_path = f"models:/{registered_model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# set the root path
root_path = Path(__file__).parent
# path of the data
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"

# model paths
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"
model_path = root_path / "models/model.joblib"

# load the objects
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
kmeans = joblib.load(kmeans_path)

# dataset to plot
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# make the title for the page
st.title("Uber Demand in New York City 🚕")

# select for only neighbors or all
st.sidebar.title("Options")
map_type = st.sidebar.radio(label="Select the type of Map",
                     options=["Complete NYC Map", "Only for Neighborhood Regions"],
                     index=1)

# select the date
date = st.date_input("Select the date", value=None,
                     min_value=dt.date(year=2016, month=3, day=1),
                     max_value=dt.date(year=2016, month=3, day=31)) 

# select the time of day
time = st.time_input("Select the time", value=None)

if date and time:

       # next time interval
       delta = dt.timedelta(minutes=15)
       next_interval = dt.datetime(year=date.year,
                                   month=date.month,
                                   day=date.day, 
                                   hour=time.hour, 
                                   minute=time.minute) + delta

       # sample a latitude longitude value randomly from the dataset
       sample_loc = df_plot.sample(1).reset_index(drop=True)
       lat = sample_loc["pickup_latitude"].item()
       long = sample_loc["pickup_longitude"].item()

       # create the datetime index
       index = pd.Timestamp(f"{date} {next_interval.time()}")
       region = sample_loc["region"].item()

       with st.spinner("Fetching your Current Location, Region, Date & Time..."):
              sleep(3)

       st.write("**Location** : ", f"({lat}, {long})")
       st.write("**Date & Time:**", index)
       st.write("Region ID: ", region)

       # scale the data
       scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])
       
       # plot the map
       st.subheader("MAP")
       
       # list of 30 hex colors on a white background with 8 digits
       colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
              "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
              "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
              "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
              "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", 
              "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]

       # add color to the data
       region_colors = {region: colors[i] for i, region in enumerate(df_plot["region"].unique().tolist())}
       df_plot["color"] = df_plot["region"].map(region_colors)
       
       # make prediction pipeline
       pipe = Pipeline([
              ('encoder', encoder),
              ('reg', model)
       ])
       


       if map_type == "Complete NYC Map":
              # progress bar
              progress_bar = st.progress(value=0, text="Operation in progress. Please wait.")
              for percent_complete in range(100):
                     sleep(0.05)
                     progress_bar.progress(percent_complete + 1, text="Demand Prediction in progress. Please wait.")
              
              # map
              st.map(data=df_plot, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
              
              # remove the progress bar
              progress_bar.empty()
              
              # filter the data 
              input_data = df.loc[index, :].sort_values("region")
              target = input_data["total_pickups"]

              # do the predictions
              predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))
              
              # Display the map legend
              st.markdown("### Map Legend")
              for ind in range(0, 30):
                     color = colors[ind]
                     demand = predictions[ind]
                     st.markdown(
                            f'<div style="display:flex; align-items:center;">'
                                   f'<div style="background-color:{color}; width:20px; height:10px; margin-right:10px;"></div>'
                                   f'<span>Region ID: {ind} & Demand: {int(demand)} {" => Current Region" if region == ind else ""}</span>'
                            f'</div>', unsafe_allow_html=True
                     )
              
       elif map_type == "Only for Neighborhood Regions":
              
              # calculate the distances from centroid
              distances = kmeans.transform(scaled_cord).values.ravel().tolist()
              distances = list(enumerate(distances))
              sorted_distances = sorted(distances, key=lambda x: x[1])[0:9]
              indexes = sorted([ind[0] for ind in sorted_distances])
              
              # filter plot data on regions
              df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]
              
              # progress bar
              progress_bar = st.progress(value=0, text="Operation in progress. Please wait.")
              for percent_complete in range(100):
                     sleep(0.05)
                     progress_bar.progress(percent_complete + 1, text="Demand Prediction in progress. Please wait.")
              
              # map
              st.map(data=df_plot_filtered, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
              
              # remove the progress bar
              progress_bar.empty()
              
              # filter the data 
              input_data = df.loc[index, :]
              input_data = input_data.loc[input_data["region"].isin(indexes), :].sort_values("region")
              target = input_data["total_pickups"]

              # do the predictions
              predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))
              
              # Display the map legend
              st.markdown("### Map Legend")
              for ind in range(0, 9):
                     color = colors[indexes[ind]]
                     demand = predictions[ind]
                     st.markdown(
                            f'<div style="display:flex; align-items:center;">'
                            f'<div style="background-color:{color}; width:20px; height:10px; margin-right:10px;"></div>'
                                   f'<span>Region ID: {indexes[ind]} & Demand: {int(demand)} {" => Current Region" if region == indexes[ind] else ""}</span>'
                            f'</div>', unsafe_allow_html=True
                     )