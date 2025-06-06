import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import box

st.set_page_config(page_title="Grid Generator for Airshed", layout="wide")
st.title("📍 Define Airshed and Generate Population Grid")

st.markdown("Draw a rectangle on the map to define your airshed boundary.")

# Default map center
center = [30.1575, 71.5249]  # Rough center of Multan

m = folium.Map(location=center, zoom_start=10)

# Add drawing controls
from folium.plugins import Draw
Draw(export=True, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'marker': False}).add_to(m)

st_map = st_folium(m, width=700, height=500, returned_objects=["last_active_drawing"])

grid_df = None
if st_map and st_map.get("last_active_drawing"):
    geom = st_map["last_active_drawing"]
    st.write("You drew a shape:", geom)
    if geom["type"] == "Feature":
        coords = geom["coordinates"][0]
        lons, lats = zip(*coords)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Generate grid (0.01° resolution)
        resolution = 0.01
        lat_points = np.arange(min_lat, max_lat, resolution)
        lon_points = np.arange(min_lon, max_lon, resolution)

        records = []
        id_counter = 1
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                cell = box(lon, lat, lon + resolution, lat + resolution)
                records.append({
                    "id": id_counter,
                    "left": lon,
                    "right": lon + resolution,
                    "top": lat + resolution,
                    "bottom": lat,
                    "row_index": i,
                    "col_index": j
                })
                id_counter += 1

        grid_df = pd.DataFrame(records)
        st.success(f"Grid generated with {len(grid_df)} cells.")
        st.dataframe(grid_df.head())

        # Display grid on map
        m_grid = folium.Map(location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2], zoom_start=11)
        for _, row in grid_df.iterrows():
            rect = [[row['bottom'], row['left']], [row['top'], row['right']]]
            folium.Rectangle(bounds=rect, color="blue", weight=1, fill=False).add_to(m_grid)
        st.subheader("🗺️ Generated Grid")
        st_folium(m_grid, width=700, height=500)

        # Allow download
        csv = grid_df.to_csv(index=False)
        st.download_button("📥 Download Grid CSV", data=csv, file_name="airshed_grid.csv", mime="text/csv")
    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
