import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import box
import tempfile
import requests
import os
import rioxarray
import xarray as xr

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
    base_url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{country_code}/{(country_code).lower()}_ppp_{year}_UNadj_constrained.tif"
    response = requests.get(base_url, stream=True)

    if response.status_code != 200:
        st.error(f"Failed to download WorldPop raster. Status: {response.status_code}")
        return None

    temp_dir = tempfile.gettempdir()
    out_path = os.path.join(temp_dir, f"worldpop_{country_code}_{year}.tif")

    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)

    return out_path

https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/PAK/pak_ppp_2020_UNadj.tif


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
                id_counter += 1

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        st.success(f"Grid generated with {len(gdf)} cells.")

        st.info("Downloading WorldPop data (~50MB)...")
        tif_path = download_worldpop()
        if not tif_path:
            st.stop()

        st.success("WorldPop data downloaded!")

        da = rioxarray.open_rasterio(tif_path).squeeze()
        da = da.rio.write_crs("EPSG:4326")

        population = []
        for geom in gdf.geometry:
            try:
                clipped = da.rio.clip([geom.__geo_interface__], gdf.crs, drop=True, all_touched=True)
                total = float(clipped.where(clipped > 0).sum().values)
            except Exception:
                total = 0
            population.append(total)

        gdf["_sum"] = population
        st.dataframe(gdf.drop(columns="geometry").head())

        m_grid = folium.Map(location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2], zoom_start=11)
        for _, row in gdf.iterrows():
            rect = [[row['bottom'], row['left']], [row['top'], row['right']]]
            folium.Rectangle(bounds=rect, color="blue", weight=1, fill=True,
                             tooltip=f"Pop: {int(row['_sum'])}").add_to(m_grid)
        st.subheader("🗺️ Grid with Population")
        st_folium(m_grid, width=700, height=500)

        csv = gdf.drop(columns="geometry").to_csv(index=False)
        st.download_button("📥 Download Population Grid CSV", data=csv, file_name="zonal_population_stats.csv", mime="text/csv")

        os.remove(tif_path)
    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
