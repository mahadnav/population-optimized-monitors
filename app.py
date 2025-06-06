import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import box
import rioxarray
from rasterstats import zonal_stats # <-- Import the new library

st.set_page_config(page_title="Grid Generator for Airshed", layout="wide")
st.title("📍 Define Airshed and Generate Population Grid with WorldPop")

st.markdown("""
Draw a rectangle on the map to define your airshed boundary. The app will generate a 0.01° x 0.01° spatial resolution grid and compute population from WorldPop.
**Note:** You will need to install `rasterstats` for this to work (`pip install rasterstats`).
""")

# --- Initial Map Display ---
center = [30.1575, 71.5249]  # Center on Multan, Pakistan
m = folium.Map(location=center, zoom_start=10)
from folium.plugins import Draw
Draw(export=False, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'marker': False, 'polyline': False}).add_to(m)
st_map = st_folium(m, width=1500, height=500, returned_objects=["last_active_drawing"])

def get_worldpop_data():
    """Handles the upload of the WorldPop GeoTIFF file."""
    uploaded_file = st.file_uploader("📂 Upload a WorldPop GeoTIFF (.tif) file", type=["tif", "tiff"])
    if not uploaded_file:
        st.warning("Please upload a raster file to continue.")
        st.stop()
    return uploaded_file

if st_map and st_map.get("last_active_drawing"):
    geom = st_map["last_active_drawing"]
    if geom and geom.get("geometry") and geom["geometry"]["type"] == "Polygon":
        coords = geom["geometry"]["coordinates"][0]
        lons, lats = zip(*coords)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # --- Grid Generation ---
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
        st.success(f"✅ Grid generated with {len(gdf)} cells.")

        tif_file = get_worldpop_data()

        # --- Population Calculation using Zonal Statistics (More Robust Method) ---
        st.info("⏳ Calculating population in each grid cell... this may take a moment.")
        
        with st.spinner('Processing...'):
            # Use the uploaded file object directly. rasterstats can handle it.
            # 'all_touched=True' includes cells that are merely touched by the polygon.
            stats = zonal_stats(gdf, tif_file, stats="sum", all_touched=True)
            
            # The result is a list of dictionaries. Extract the 'sum' value from each.
            # Handle cases where a polygon might not overlap with the raster (results in None)
            population_sums = [stat['sum'] if stat['sum'] is not None else 0 for stat in stats]
            gdf["population"] = population_sums

        st.success("✅ Population values computed.")
        st.dataframe(gdf.drop(columns="geometry").head())

        # --- Displaying the Grid with Population Data ---
        m_grid = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=10)
        
        # Add a choropleth layer for better visualization
        folium.Choropleth(
            geo_data=gdf,
            name='choropleth',
            data=gdf,
            columns=['id', 'population'],
            key_on='feature.id', # This needs to match an id in the geo_data
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Population Count'
        ).add_to(m_grid)

        folium.LayerControl().add_to(m_grid)

        st.subheader("🗺️ Grid with Population")
        st_folium(m_grid, width=1500, height=500)

        # --- Download Functionality ---
        csv = gdf.drop(columns="geometry").to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Population Grid CSV",
            data=csv,
            file_name="zonal_population_stats.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")