import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium # type: ignore
from shapely.geometry import box
from rasterstats import zonal_stats
import matplotlib.cm as cm
import matplotlib.colors as colors
import plotly.express as px
from branca.element import Element
import json
import base64
import os
import tempfile
from geopy.distance import geodesic
from helpers.utils import classify_population_density, randomize_initial_cluster, weighted_kmeans

# --- Page Configuration ---
st.set_page_config(page_title="Population-Centric Monitoring Network", layout="wide")

# --- Custom CSS ---
# Ensure you have a 'style.css' file or remove this block
try:
    with open("style.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("style.css file not found. App will run with default styling.")

# --- Session State Management ---
def init_session_state():
    """Initializes all session state variables if they don't exist."""
    defaults = {
        "boundary": None, "grid_gdf": None, "population_grid": None, "monitor_data": None,
        "last_drawn_boundary": None, "airshed_confirmed": False, "population_computed": False, "bounds": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_analysis():
    """Explicitly resets session state variables for a new analysis run."""
    init_session_state() # Ensure all keys exist before trying to set them
    st.session_state.boundary = None
    st.session_state.grid_gdf = None
    st.session_state.population_grid = None
    st.session_state.monitor_data = None
    st.session_state.airshed_confirmed = False
    st.session_state.population_computed = False
    st.session_state.bounds = None

init_session_state()

# --- Helper Functions ---
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def merge_close_centroids(centroids, threshold=2):
    if centroids.empty:
        return centroids
    merged_centroids, used = [], set()
    for i, row1 in centroids.iterrows():
        if i in used: continue
        close_centroids = [row1]
        for j, row2 in centroids.iterrows():
            if i != j and j not in used:
                if calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon'])) < threshold:
                    close_centroids.append(row2)
                    used.add(j)
        if len(close_centroids) > 1:
            mean_lat = np.mean([c['lat'] for c in close_centroids])
            mean_long = np.mean([c['lon'] for c in close_centroids])
            merged_centroids.append({'lat': mean_lat, 'lon': mean_long})
        else:
            merged_centroids.append({'lat': row1['lat'], 'lon': row1['lon']})
        used.add(i)
    new_centroids = pd.DataFrame(merged_centroids)
    for i, row1 in new_centroids.iterrows():
        for j, row2 in new_centroids.iterrows():
            if i != j and calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon'])) < threshold:
                return merge_close_centroids(new_centroids, threshold)
    return new_centroids

# --- UI Layout ---

# --- Header ---
col1, col2 = st.columns([5, 1], vertical_alignment="center")
with col1:
    st.title("Population-Centric Air Quality Monitor Optimization")
    st.markdown("### A tool to strategically place air quality monitors based on population density.")
    st.write("Developed by Mahad Naveed and the PakAirQuality (PAQI) Team.")
    st.write("[doi.org/10.5194/egusphere-egu25-4723](https://doi.org/10.5194/egusphere-egu25-4723)")
with col2:
    try:
        with open("logo.jpeg", "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{encoded}" width="200" style="pointer-events: none;">'
        st.markdown(logo_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("logo.jpeg not found.")
st.divider()

# --- STEP 1: DEFINE AIRSHED ---
st.markdown("#### Step 1: Define Your Airshed")
m = folium.Map(zoom_start=8, tiles="CartoDB positron")
folium.plugins.Draw(export=False, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'circlemarker': False, 'marker': False, 'polyline': False}).add_to(m)
st_map = st_folium(m, width=1700, height=500, returned_objects=["last_active_drawing"])

# --- Drawing Detection & State Reset Logic ---
if st_map and st_map.get("last_active_drawing"):
    new_drawing = st_map["last_active_drawing"]
    if new_drawing != st.session_state.last_drawn_boundary:
        reset_analysis()
        st.session_state.last_drawn_boundary = new_drawing
        st.rerun()

# --- Confirmation Button ---
if st.session_state.last_drawn_boundary and not st.session_state.airshed_confirmed:
    st.warning("An airshed has been drawn. Please confirm to proceed.")
    _, col2, _ = st.columns([2, 1, 2])
    with col2:
        if st.button("✅ Confirm Airshed and Proceed", use_container_width=True):
            st.session_state.boundary = st.session_state.last_drawn_boundary
            st.session_state.airshed_confirmed = True
            st.rerun()

# --- Main Analysis Workflow ---
if st.session_state.airshed_confirmed:
    st.markdown("---")
    st.markdown("#### Step 2: Generate Grid & Upload Population Data")

    if st.session_state.grid_gdf is None:
        with st.spinner("Generating analysis grid..."):
            geom = st.session_state.boundary
            coords = geom["geometry"]["coordinates"][0]
            lons, lats = zip(*coords)
            min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
            st.session_state.bounds = {'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}
            resolution = 0.01
            lat_points, lon_points = np.arange(min_lat, max_lat, resolution), np.arange(min_lon, max_lon, resolution)
            records = [{"id": i * len(lon_points) + j + 1, "geometry": box(lon, lat, lon + resolution, lat + resolution)} for i, lat in enumerate(lat_points) for j, lon in enumerate(lon_points)]
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
            st.session_state.grid_gdf = gdf
            st.success(f"Grid generated with {len(gdf)} cells.")

    tif_file = st.file_uploader("📂 Upload a WorldPop GeoTIFF (.tif) file", type=["tif", "tiff"])
    st.write("Sample data: [WorldPop GeoTIFF United Kingdom](https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/GBR/gbr_ppp_2020_UNadj.tif)")

    if tif_file and not st.session_state.population_computed:
        st.markdown("---")
        st.markdown("#### Step 3: Run Population Analysis")
        st.info("The grid and population data are ready. Click the button to start the calculation.")
        _, col2, _ = st.columns([2, 1, 2])
        with col2:
            if st.button("Calculate Population Density", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(tif_file.getvalue())
                    tmp_path = tmp.name
                
                gdf = st.session_state.grid_gdf
                total_geometries, chunk_size, population_sums = len(gdf), 500, []
                progress_bar = st.progress(0, text="Initializing...")
                for i in range(0, total_geometries, chunk_size):
                    chunk_gdf = gdf.iloc[i:i + chunk_size]
                    stats = zonal_stats(chunk_gdf, tmp_path, stats="sum", all_touched=True, geojson_out=False)
                    population_sums.extend([stat['sum'] if stat and stat['sum'] is not None else 0 for stat in stats])
                    processed_count = min(i + chunk_size, total_geometries)
                    percent_complete = processed_count / total_geometries
                    progress_bar.progress(percent_complete, text=f"Processing... {processed_count}/{total_geometries} cells ({percent_complete:.0%})")
                
                progress_bar.empty()
                os.remove(tmp_path)
                gdf["population"] = population_sums
                st.session_state.population_grid = gdf[gdf['population'] > 0].copy().reset_index(drop=True)
                st.session_state.population_computed = True
                st.success("✅ Population analysis complete!")

    if st.session_state.population_computed:
        st.markdown("---")
        st.markdown("#### Step 4: Review Population Data")
        gdf = st.session_state.population_grid
        bounds = st.session_state.bounds
        tab1, tab2, tab3 = st.tabs(["🗺️ Population Map", "📊 Population Distribution", "📥 Download Data"])

        with tab1:
            st.subheader("Population Heatmap")
            density_categories = [
                {'label': 'Very Low', 'min': 0, 'max': 100, 'color': '#fee5d9', 'range_text': '0 - 100'},
                {'label': 'Low', 'min': 101, 'max': 500, 'color': '#fcbba1', 'range_text': '101 - 500'},
                # ... Add other categories as needed
                {'label': 'Extreme', 'min': 10001, 'max': float('inf'), 'color': '#a50f15', 'range_text': '10,001+'}
            ]
            def get_color(pop):
                for cat in density_categories:
                    if cat['min'] <= pop <= cat['max']: return cat['color']
                return '#808080'
            def style_function(feature):
                return {'fillColor': get_color(feature['properties'].get('population', 0)), 'color': 'none', 'weight': 0, 'fillOpacity': 0.75}
            
            map_center = [(bounds['min_lat'] + bounds['max_lat']) / 2, (bounds['min_lon'] + bounds['max_lon']) / 2]
            m_grid = folium.Map(location=map_center, zoom_start=8)
            folium.GeoJson(json.loads(gdf.to_json()), name='Population Grid', style_function=style_function, tooltip=folium.GeoJsonTooltip(fields=['population'], aliases=['Population:'])).add_to(m_grid)
            # Add Legend
            headers, ranges = ''.join([f'<div style="flex: 1; text-align: center; padding: 4px; background-color:{c["color"]}; border: 1px solid #333; color: white;">{c["label"]}</div>' for c in density_categories]), ''.join([f'<div style="flex: 1; text-align: center; padding: 4px; border: 1px solid #333;">{c["range_text"]}</div>' for c in density_categories])
            legend_html = f'''<div style="position: fixed; bottom: 20px; left: 20px; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.9); padding: 5px; font-size: 14px; font-family: Arial, sans-serif; min-width: 500px;"><div style="display: flex; flex-direction: row; font-weight: bold;">{headers}</div><div style="display: flex; flex-direction: row;">{ranges}</div></div>'''
            m_grid.get_root().html.add_child(Element(legend_html))
            st_folium(m_grid, width=1700, height=500)

        with tab2:
            st.subheader("Population Count Distribution")
            density_df = classify_population_density(gdf.copy())
            fig = px.histogram(density_df, x="population", color="Density", nbins=250, marginal="rug", barmode='overlay')
            fig.update_traces(opacity=0.75)
            fig.update_layout(xaxis_title='Population Count per Cell', yaxis_title='Count of Grid Cells', legend_title='Density Level')
            st.plotly_chart(fig)

        with tab3:
            st.subheader("Download Processed Population Data")
            csv = gdf.to_csv(index=False).encode('utf-8')
            _, col2, _ = st.columns([2, 1, 2])
            with col2:
                st.download_button("📥 Download Population Grid CSV", data=csv, file_name="zonal_population_stats.csv", mime="text/csv", use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Step 5: Configure and Run Monitor Optimization")
        density_df['long'] = density_df.geometry.centroid.x
        density_df['lat'] = density_df.geometry.centroid.y
        
        col1, col2, col3 = st.columns(3)
        with col1: low_monitors = st.number_input("Monitors for Low Density", min_value=1, value=5)
        with col2: high_monitors = st.number_input("Monitors for High Density", min_value=1, value=10)
        with col3: min_dist = st.number_input("Min Distance Between Monitors (km)", min_value=1, value=2)

        _, col2, _ = st.columns([2, 1.5, 2])
        with col2:
            if st.button("Optimize Monitoring Network", type="primary", use_container_width=True):
                with st.spinner("Optimizing monitor locations..."):
                    vals = density_df[['population', 'long', 'lat', 'Density']].copy()
                    low, high = vals[vals['Density'] == 'Low'], vals[vals['Density'] == 'High']
                    low_df, high_df = pd.DataFrame(), pd.DataFrame()
                    
                    if not low.empty and low_monitors > 0:
                        _, centers_low, _, _ = weighted_kmeans(low, randomize_initial_cluster(low, low_monitors), low_monitors)
                        low_df = pd.DataFrame(centers_low, columns=['lon', 'lat'])[['lat', 'lon']]
                    if not high.empty and high_monitors > 0:
                        _, centers_high, _, _ = weighted_kmeans(high, randomize_initial_cluster(high, high_monitors), high_monitors)
                        high_df = pd.DataFrame(centers_high, columns=['lon', 'lat'])[['lat', 'lon']]

                    raw_df = pd.concat([low_df, high_df], ignore_index=True)
                    st.session_state.monitor_data = merge_close_centroids(raw_df, threshold=min_dist)
                    st.success("✅ Optimization complete!")

    if st.session_state.monitor_data is not None:
        st.markdown("---")
        st.markdown("#### Step 6: Review Final Optimized Monitor Locations")
        final_df = st.session_state.monitor_data
        tab1, tab2 = st.tabs(["🗺️ Final Monitor Map", "📥 Download Locations"])
        with tab1:
            map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
            m_final = folium.Map(location=map_center, zoom_start=10)
            if st.session_state.get("boundary"):
                folium.GeoJson(st.session_state.boundary, style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0.0}, name='Airshed Boundary').add_to(m_final)
            for index, row in final_df.iterrows():
                folium.CircleMarker(location=[row['lat'], row['lon']], radius=4, color='#e63946', fill=True, fill_color='#e63946', popup=f"Monitor #{index+1}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}").add_to(m_final)
            st_folium(m_final, width=1700, height=700)
        with tab2:
            st.dataframe(final_df.style.format({'lat': '{:.5f}', 'lon': '{:.5f}'}))
            _, col2, _ = st.columns([2, 1, 2])
            with col2:
                st.download_button("📥 Download Monitor Locations CSV", data=final_df.to_csv(index=False).encode('utf-8'), file_name="optimized_monitor_locations.csv", mime="text/csv", use_container_width=True)