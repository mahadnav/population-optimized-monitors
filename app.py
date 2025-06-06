import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import box
from rasterstats import zonal_stats
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import json
import os

from helpers import utils

st.set_page_config(
    page_title="Population Optimized Monitors",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------------------------------------------------

# 3. Now the rest of your app can run
st.title("Population Optimized Monitors")
st.write("Welcome to the application.")
st.write(utils.test_function())