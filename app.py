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
from helpers.utils import classify_population_density

if "population_grid" not in st.session_state:
    st.session_state["population_grid"] = None
if "population_computed" not in st.session_state:
    st.session_state["population_computed"] = False


st.set_page_config(page_title="Grid Generator for Airshed", layout="wide")
st.title("📍 Define Airshed and Generate Population Grid with WorldPop")

st.markdown("""
Draw a rectangle on the map to define your airshed boundary.
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

        # --- Population Calculation Only If Not Already Done ---
        if not st.session_state["population_computed"]:
            st.info("⏳ Calculating population in each grid cell...")
            total_geometries = len(gdf)
            chunk_size = 100
            population_sums = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(0, total_geometries, chunk_size):
                chunk_gdf = gdf.iloc[i:i + chunk_size]
                stats = zonal_stats(chunk_gdf, tif_file, stats="sum", all_touched=True)
                chunk_sums = [stat['sum'] if stat and stat['sum'] is not None else 0 for stat in stats]
                population_sums.extend(chunk_sums)

                processed_count = min(i + chunk_size, total_geometries)
                percent_complete = processed_count / total_geometries
                progress_bar.progress(percent_complete)
                status_text.text(f"Processing... {processed_count}/{total_geometries} cells complete ({percent_complete:.0%})")

            status_text.text(f"Processed {total_geometries} cells.")
            progress_bar.empty()
            gdf["population"] = population_sums

            st.success("✅ Population values computed.")
            st.session_state["population_grid"] = gdf
            st.session_state["population_computed"] = True
        else:
            gdf = st.session_state["population_grid"]
            st.success("✅ Population values retrieved from session.")

        st.success("✅ Population values computed.")

        # --- Displaying the Grid with Population Data ---
        m_grid = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=8)
        
        # Convert gdf to GeoJSON and assign feature ids as string matching gdf 'id'
        gdf = gdf.fillna(0).reset_index(drop=True)
        gdf['str_id'] = gdf['id'].astype(str)
        geojson_data = json.loads(gdf.to_json())

        for i, feature in enumerate(geojson_data['features']):
            feature['id'] = gdf.loc[i, 'str_id']

        # Create colormap for population values
        pop_min = gdf['population'].min()
        pop_max = gdf['population'].max()
        colormap = cm.get_cmap('plasma')

        norm = colors.Normalize(vmin=pop_min, vmax=pop_max)

        def style_function(feature):
            pop = gdf.loc[gdf['str_id'] == feature['id'], 'population'].values[0]
            if pop == 0 or pop is None or np.isnan(pop):
                # Fully transparent for zero population
                return {
                    'fillOpacity': 0,
                    'weight': 0
                }
            else:
                color = colors.rgb2hex(colormap(norm(pop))[:3])
                return {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                }

        folium.GeoJson(
            geojson_data,
            name='Population Grid',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['population'], aliases=['Population:'])
        ).add_to(m_grid)

        folium.LayerControl().add_to(m_grid)

        csv = gdf.to_csv(index=False).encode('utf-8')

        st.subheader("🗺️ Grid with Population")
        st_folium(m_grid, width=1500, height=500)


        gdf = gdf.drop(columns=["geometry"])
        gdf['long'], gdf['lat'] = (gdf['left']+ gdf['right'])/2, (gdf['top'] + gdf['bottom'])/2
        gdf.drop(columns=['left', 'right', 'top', 'bottom'], inplace=True)
        gdf.fillna(0, inplace=True)
        gdf = gdf.loc[~(gdf['population']==0)].reset_index(drop=True)
        gdf = gdf[["id", 'long', 'lat', 'row_index', 'col_index', 'population']]
        csv = gdf.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Population Grid CSV",
            data=csv,
            file_name="zonal_population_stats.csv",
            mime="text/csv"
        )

        st.subheader("Population Density Classification")
        density_df = classify_population_density(gdf.copy())
        st.dataframe(density_df)
    

    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
