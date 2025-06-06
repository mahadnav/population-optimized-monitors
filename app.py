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
Draw(export=False, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'marker': False}).add_to(m)
st_map = st_folium(m, width=1500, height=500, returned_objects=["last_active_drawing"])

def get_worldpop(country_code="PAK", year="2020"):
    # base_url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{country_code}/{country_code.lower()}_ppp_2020_UNadj.tif"
    # response = requests.get(base_url, stream=True)

    # if response.status_code != 200:
    #     st.error("Failed to download WorldPop raster.")
    #     return None

    # total_size = int(response.headers.get('content-length', 0))
    # total_mb = total_size / (1024 * 1024)  # convert bytes to megabytes

    # temp_dir = tempfile.gettempdir()
    # out_path = os.path.join(temp_dir, f"worldpop_{country_code}_{year}.tif")

    # chunk_size = 1024
    # downloaded = 0
    # progress_bar = st.progress(0)
    # status_text = st.empty()

    # with open(out_path, 'wb') as f:
    #     for chunk in response.iter_content(chunk_size):
    #         if chunk:
    #             f.write(chunk)
    #             downloaded += len(chunk)
    #             percent = int((downloaded / total_size) * 100)
    #             downloaded_mb = downloaded / (1024 * 1024)
    #             status_text.text(f"⬇️ Downloaded: {downloaded_mb:.2f} MB / {total_mb:.2f} MB ({percent}%)")
    #             progress_bar.progress(min(percent, 100))

    # progress_bar.empty()
    # status_text.success("✅ Download complete.")
    # return out_path

    uploaded_file = st.file_uploader("📂 Upload a WorldPop GeoTIFF (.tif) file", type=["tif", "tiff"])

    if not uploaded_file:
        st.warning("Please upload a raster file to continue.")
        st.stop()
    return uploaded_file


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

        st.info("Downloading WorldPop data...")
        tif_path = get_worldpop()
        st.write(f"WorldPop data saved to: {tif_path}")
        if not tif_path:
            st.stop()

        st.success("WorldPop data downloaded!")

        da = rioxarray.open_rasterio(tif_path).squeeze()
        da = da.rio.write_crs("EPSG:4326")

        population = []
        for i, geom in enumerate(gdf.geometry):
            try:
                clipped = da.rio.clip([geom.__geo_interface__], gdf.crs, drop=True, all_touched=True)
                st.write(clipped)
                # Guard against fully masked array
                total = float(clipped.where(clipped.notnull()).sum().values)
            except Exception as e:
                st.warning(f"Failed to clip geometry {i}: {e}")
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
