import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import box
from rasterio.mask import mask
import rasterio
import tempfile
import requests
import os
import zipfile

st.set_page_config(page_title="Grid Generator for Airshed", layout="wide")
st.title("📍 Define Airshed and Generate Population Grid with WorldPop")

st.markdown("""
Draw a rectangle on the map to define your airshed boundary. The app will generate a 0.01° x 0.01° spatial resolution grid and compute population from WorldPop.
""")

center = [30.1575, 71.5249]  # Center on Multan
m = folium.Map(location=center, zoom_start=10)
from folium.plugins import Draw
Draw(export=True, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'marker': False}).add_to(m)
st_map = st_folium(m, width=700, height=500, returned_objects=["last_active_drawing"])

def download_worldpop(country_code="PAK", year="2020"):
    base_url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{country_code}/ppp_{year}_{country_code}_1km_Aggregated.tif"
    response = requests.get(base_url, stream=True)
    temp_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    with open(temp_tif.name, 'wb') as out:
        out.write(response.content)
    return temp_tif.name

grid_df = None
if st_map and st_map.get("last_active_drawing"):
    geom = st_map["last_active_drawing"]
    if geom["type"] == "Feature":
        coords = geom["geometry"]["coordinates"][0]
        lons, lats = zip(*coords)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        resolution = 0.01
        lat_points = np.arange(min_lat, max_lat, resolution)
        lon_points = np.arange(min_lon, max_lon, resolution)

        records = []
        id_counter = 1
        grid_geoms = []
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                geom_box = box(lon, lat, lon + resolution, lat + resolution)
                records.append({
                    "id": id_counter,
                    "left": lon,
                    "right": lon + resolution,
                    "top": lat + resolution,
                    "bottom": lat,
                    "row_index": i,
                    "col_index": j,
                    "geometry": geom_box
                })
                grid_geoms.append(geom_box)
                id_counter += 1

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        st.success(f"Grid generated with {len(gdf)} cells.")

        # Download WorldPop and crop to bounds
        st.info("Downloading WorldPop data (~50MB)...")
        tif_path = download_worldpop()
        st.success("WorldPop data downloaded!")

        population = []
        with rasterio.open(tif_path) as src:
            for geom in gdf.geometry:
                try:
                    out_image, _ = mask(src, [geom.__geo_interface__], crop=True, all_touched=True)
                    total = out_image[out_image > 0].sum() if out_image.size > 0 else 0
                except Exception:
                    total = 0
                population.append(total)

        gdf["_sum"] = population
        st.dataframe(gdf.drop(columns="geometry").head())

        # Map
        m_grid = folium.Map(location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2], zoom_start=11)
        for _, row in gdf.iterrows():
            rect = [[row['bottom'], row['left']], [row['top'], row['right']]]
            folium.Rectangle(bounds=rect, color="blue", weight=1, fill=True,
                             tooltip=f"Pop: {int(row['_sum'])}").add_to(m_grid)
        st.subheader("🗺️ Grid with Population")
        st_folium(m_grid, width=700, height=500)

        # Download CSV
        csv = gdf.drop(columns="geometry").to_csv(index=False)
        st.download_button("📥 Download Population Grid CSV", data=csv, file_name="zonal_population_stats.csv", mime="text/csv")

        # Cleanup
        os.remove(tif_path)
    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
