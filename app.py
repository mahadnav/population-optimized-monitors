import streamlit as st
from streamlit_extras.stylable_container import stylable_container # type: ignore
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium # type: ignore

from shapely.geometry import box
from rasterstats import zonal_stats

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import branca.colormap as bcm
from branca.element import Element

import json
import base64
import os
import time
import tempfile
from geopy.distance import geodesic
from helpers.utils import classify_population_density, randomize_initial_cluster, weighted_kmeans

# --- Page Configuration ---
st.set_page_config(page_title="Population-Centric Monitoring Network", layout="wide")

# --- Custom CSS ---
with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# --- Session State Initialization ---
# Ensures variables persist across reruns and are properly initialized
def init_session_state():
    defaults = {
        "boundary": None,
        "grid_gdf": None,
        "population_grid": None,
        "monitor_data": None,
        "last_drawn_boundary": None,
        "airshed_confirmed": False,
        "population_computed": False,
        "bounds": None,
        "cached_raster": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_analysis():
    """Explicitly resets all session state variables for a new analysis run."""
    st.session_state.boundary = None
    st.session_state.grid_gdf = None
    st.session_state.population_grid = None
    st.session_state.monitor_data = None
    st.session_state.last_drawn_boundary = None # Also reset the last drawing
    st.session_state.airshed_confirmed = False
    st.session_state.population_computed = False
    st.session_state.bounds = None

def add_tile_layers(folium_map):
    """Adds multiple popular tile layer options to a Folium map."""
    
    # Add the default tile layer which is light and clean
    folium.TileLayer('CartoDB positron', name='Light Mode', control=True).add_to(folium_map)
    
    # Add a dark mode tile layer
    folium.TileLayer('CartoDB dark_matter', name='Dark Mode', control=True).add_to(folium_map)
    
    # Add a standard street map
    folium.TileLayer('OpenStreetMap', name='Street Map', control=True).add_to(folium_map)

    # Add the layer control panel to the map, allowing users to switch
    folium.LayerControl().add_to(folium_map)

init_session_state()

# --- Helper Functions ---
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def merge_close_centroids(centroids, threshold=2):
    # This recursive function correctly merges centroids that are close to each other
    if centroids.empty:
        return centroids
    merged_centroids = []
    used = set()
    for i, row1 in centroids.iterrows():
        if i in used: continue
        close_centroids = [row1]
        for j, row2 in centroids.iterrows():
            if i != j and j not in used:
                distance = calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon']))
                if distance < threshold:
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
    # Check if a recursive merge is needed
    for i, row1 in new_centroids.iterrows():
        for j, row2 in new_centroids.iterrows():
            if i != j:
                if calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon'])) < threshold:
                    return merge_close_centroids(new_centroids, threshold)
    return new_centroids

# --- Header ---
col1, col2 = st.columns([5, 1], vertical_alignment="center")
with col1:
    st.title("Population-Centric Air Quality Monitor Optimization")
    st.markdown("### A tool to strategically place air quality monitors based on population density.")
    st.write("Developed by Mahad Naveed and the PakAirQuality (PAQI) Team.")
    st.write("[doi.org/10.5194/egusphere-egu25-4723](https://doi.org/10.5194/egusphere-egu25-4723)")
with col2:
    with open("logo.jpeg", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    logo_html = f'<img src="data:image/jpeg;base64,{encoded}" width="200" style="pointer-events: none;">'
    st.markdown(logo_html, unsafe_allow_html=True)
st.divider()

_, _, col3 = st.columns([3, 3, 1], vertical_alignment="center")
with col3:
    with stylable_container(
        "blue_button_no_hover",  # It's good practice to give a unique key
        css_styles="""
        button {
            background-color: #3153a5;
            color: white;
            border: 1px solid #3153a5;
        }

        button:hover {
            background-color: #3153a5 !important;
            color: white !important;
            border-color: #3153a5 !important;
        }
        
        button:active {
            background-color: #3153a5 !important;
            color: white !important;
            border-color: #3153a5 !important;
        }
        """,
    ):
        button1 = st.button("🔄 Reset", key="reset")
    if button1:
        reset_analysis()
        st.rerun()

# --- STEP 1: DEFINE AIRSHED ---
st.markdown("#### Define Your Airshed")
m = folium.Map(zoom_start=5, tiles=None) # Set tiles=None initially
add_tile_layers(m)
Draw(export=False, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'circlemarker': False, 'marker': False, 'polyline': False}).add_to(m)
st_map = st_folium(m, use_container_width=True, returned_objects=["last_active_drawing"])

# --- Logic to detect a new drawing and require confirmation ---
if st_map and st_map.get("last_active_drawing"):
    st.session_state.last_drawn_boundary = st_map["last_active_drawing"]

# --- Confirmation Button ---
if st.session_state.last_drawn_boundary and not st.session_state.airshed_confirmed:
    st.warning("An airshed has been drawn. Please confirm to proceed.")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("✅ Confirm Airshed and Proceed"):
            st.session_state.boundary = st.session_state.last_drawn_boundary
            st.session_state.airshed_confirmed = True
            st.rerun()

# --- STEP 2: GENERATE GRID & UPLOAD DATA ---
if st.session_state.airshed_confirmed:
    st.markdown("#### Generate Grid & Upload Population Data")

    if st.session_state.grid_gdf is None:
        with st.spinner("Generating analysis grid for the selected airshed..."):
            geom = st.session_state.boundary
            coords = geom["geometry"]["coordinates"][0]
            lons, lats = zip(*coords)
            min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
            st.session_state.bounds = {'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}

            resolution = 0.01
            lat_points, lon_points = np.arange(min_lat, max_lat, resolution), np.arange(min_lon, max_lon, resolution)
            records = [{"id": i * len(lon_points) + j + 1, "geometry": box(lon, lat, lon + resolution, lat + resolution)}
                       for i, lat in enumerate(lat_points) for j, lon in enumerate(lon_points)]
            
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
            st.session_state.grid_gdf = gdf
            st.success(f"Grid generated with {len(gdf)} cells.")

    if st.session_state.cached_raster:
        st.success(f"Using cached raster file: **{st.session_state.cached_raster['name']}**")

    tif_file = st.file_uploader(
        "Upload a new WorldPop GeoTIFF to replace the cached file",
        type=["tif", "tiff"]
    )

    if tif_file:
        with st.spinner(f"Caching new raster file: {tif_file.name}..."):
            st.session_state.cached_raster = {
                "name": tif_file.name,
                "bytes": tif_file.getvalue()
            }
    st.write("Sample data: [WorldPop GeoTIFF United Kingdom](https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/GBR/gbr_ppp_2020_UNadj.tif)")

    # --- STEP 3: RUN POPULATION ANALYSIS ---
    if st.session_state.cached_raster and not st.session_state.population_computed:
        st.markdown("#### Run Population Analysis")
        st.info("Click the button to compute zonal statistics.")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Calculate Population Density", type="primary"):
                gdf = st.session_state.grid_gdf
                # Use the cached file's bytes for the analysis
                raster_bytes = st.session_state.cached_raster['bytes']
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(raster_bytes)
                    tmp_path = tmp.name
                    
                    # --- Progress Bar and Chunking Logic ---
                    total_geometries = len(gdf)
                    chunk_size = 10
                    population_sums = []

                    # The progress bar can also contain text
                    progress_bar = st.progress(0, text="Initializing...")

                    for i in range(0, total_geometries, chunk_size):
                        chunk_gdf = gdf.iloc[i:i + chunk_size]
                        
                        # Run stats on the chunk
                        stats = zonal_stats(chunk_gdf, tmp_path, stats="sum", all_touched=True, geojson_out=False)
                        
                        chunk_sums = [stat['sum'] if stat and stat['sum'] is not None else 0 for stat in stats]
                        population_sums.extend(chunk_sums)

                        # --- Update Progress Bar ---
                        processed_count = min(i + chunk_size, total_geometries)
                        percent_complete = processed_count / total_geometries
                        progress_text = f"Processing... {processed_count}/{total_geometries} cells complete ({percent_complete:.0%})"
                        progress_bar.progress(percent_complete, text=progress_text)

                    # --- Finalize and Clean Up ---
                    progress_bar.empty()
                    os.remove(tmp_path)
                    
                    # Assign the calculated sums to the dataframe
                    st.dataframe(gdf.head())  # Show sample data before population assignment
                    gdf["population"] = np.round(gdf["population"])
                    
                    st.session_state.population_grid = gdf[gdf['population'] > 0].copy().reset_index(drop=True)
                    st.session_state.population_computed = True
                    
                    st.success("✅ Population analysis complete!")
                    import time
                    time.sleep(1)
                    st.rerun()

    # --- STEP 4: REVIEW POPULATION DATA ---
    if st.session_state.population_computed:
        gdf = st.session_state.population_grid
        bounds = st.session_state.bounds
        tab1, tab2, tab3 = st.tabs(["Population Map", "Population Distribution", "Download Grid Data"])

        with tab1:
            st.subheader("Population Heatmap")

            map_gdf = gdf.copy()
            map_gdf['population'] = pd.to_numeric(map_gdf['population'], errors='coerce').fillna(0)

            density_categories = [
            {'label': 'Very Low',  'min': 0,      'max': 10,    'color': '#fee5d9', 'range_text': '0 - 10'},
            {'label': 'Low',       'min': 11,    'max': 100,    'color': '#fcbba1', 'range_text': '10 - 100'},
            {'label': 'Moderate',  'min': 101,    'max': 500,   'color': '#fc9272', 'range_text': '101 - 500'},
            {'label': 'High',      'min': 501,   'max': 2000,   'color': '#fb6a4a', 'range_text': '500 - 2,000'},
            {'label': 'Very High', 'min': 2001,   'max': 5000,  'color': '#de2d26', 'range_text': '2,000 - 5,000'},
            {'label': 'Extreme',   'min': 5001,  'max': float('inf'), 'color': '#a50f15', 'range_text': '5,000+'}
        ]

        # Helper function to get the color for a given population value
        def get_color_for_population(population):
            for category in density_categories:
                if category['min'] <= population <= category['max']:
                    return category['color']
            return '#808080'

        # 2. STYLE FUNCTION FOR THE MAP
        def style_function(feature):
            pop = feature['properties'].get('population', 0)
            return {
                'fillColor': get_color_for_population(pop),
                'color': 'transparent',
                'weight': 0,
                'fillOpacity': 0.6
            }

        # --- Map Creation ---
        bounds = st.session_state.bounds
        map_center = [(bounds['min_lat'] + bounds['max_lat']) / 2, (bounds['min_lon'] + bounds['max_lon']) / 2]
        m_grid = folium.Map(location=map_center, zoom_start=8, tiles=None)

        # Use the cleaned map_gdf to create the GeoJSON
        geojson_data = json.loads(map_gdf.to_json())

        folium.GeoJson(
            geojson_data,
            name='Population Grid',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['population'], aliases=['Population:']),
            show=True
        ).add_to(m_grid)


        # 3. BUILD AND ADD THE CUSTOM HTML LEGEND
        # This HTML and CSS code creates the horizontal legend like your example.
        legend_header_parts, legend_range_parts = [], []

        for category in density_categories:
            header = f'<div style="flex: 1; text-align: center; padding: 4px; background-color:{category["color"]}; border: 1px solid #333; color: white;">{category["label"]}</div>'
            range_text = f'<div style="flex: 1; text-align: center; padding: 4px; border: 1px solid #333;">{category["range_text"]}</div>'
            legend_header_parts.append(header)
            legend_range_parts.append(range_text)

        legend_html = f'''
            <div style="position: fixed; 
                        bottom: 20px; left: 20px; 
                        z-index:9999; 
                        border:1px solid grey; 
                        background-color:rgba(255, 255, 255, 1);
                        padding: 5px;
                        font-size: 12px;
                        font-family: Roboto, sans-serif;
                        min-width: 500px;
                        ">
                <div style="display: flex; flex-direction: row; font-weight: bold; justify-content: space-around;">
                    {''.join(legend_header_parts)}
                </div>
                <div style="display: flex; flex-direction: row; justify-content: space-around;">
                    {''.join(legend_range_parts)}
                </div>
            </div>
        '''

        # Add the custom legend to the map
        
        m_grid.get_root().html.add_child(Element(legend_html))
        add_tile_layers(m_grid)  # Add tile layers to the map
        pop_map = st_folium(m_grid, use_container_width=True)

        with tab2:
            st.subheader("Population Count Distribution")
            density_df = classify_population_density(gdf.copy())

            # Create a true histogram where the y-axis is the count
            fig = px.histogram(
                density_df,
                x="population",
                color="Density", # This automatically creates 'Low' and 'High' groups
                nbins=250,
                marginal="rug",  # Adds the rug plot at the bottom, like in your original image
                barmode='overlay' # Overlays the histograms
            )

            # Make the overlaid bars slightly transparent to see both
            fig.update_traces(opacity=0.75)

            # Update the layout for clarity
            fig.update_layout(
                xaxis_title='Population Count per Cell',
                yaxis_title='Count of Grid Cells',
                legend_title='Density Level'
            )

            st.plotly_chart(fig)

        with tab3:
            st.subheader("Download Processed Population Data")
            csv = gdf.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Population Grid CSV", data=csv, file_name="zonal_population_stats.csv", mime="text/csv")
        
        # --- STEP 5: CONFIGURE & RUN OPTIMIZATION ---
        st.markdown("#### Configure and Run Monitor Optimization")
        density_df['long'] = density_df.geometry.centroid.x
        density_df['lat'] = density_df.geometry.centroid.y
        
        col1, col2, col3 = st.columns(3)
        with col1:
            low_monitors = st.number_input("Monitors for Low Density", min_value=1, value=5)
        with col2:
            high_monitors = st.number_input("Monitors for High Density", min_value=1, value=10)
        with col3:
            min_dist = st.number_input("Min Distance Between Monitors (km)", min_value=1, value=2)

        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            if st.button("Optimize Monitoring Network", type="primary"):
                with st.spinner("Optimizing monitor locations..."):
                    vals = density_df[['population', 'long', 'lat', 'Density']].copy()
                    low = vals[vals['Density'] == 'Low']
                    high = vals[vals['Density'] == 'High']
                    
                    low_df, high_df = pd.DataFrame(), pd.DataFrame()
                    if not low.empty and low_monitors > 0:
                        _, centers_low, _, _ = weighted_kmeans(low, randomize_initial_cluster(low, low_monitors), low_monitors)
                        low_df = pd.DataFrame([{'lat': c['coords'][1], 'lon': c['coords'][0]} for c in centers_low])
                    if not high.empty and high_monitors > 0:
                        _, centers_high, _, _ = weighted_kmeans(high, randomize_initial_cluster(high, high_monitors), high_monitors)
                        high_df = pd.DataFrame([{'lat': c['coords'][1], 'lon': c['coords'][0]} for c in centers_high])

                    raw_df = pd.concat([low_df, high_df], ignore_index=True)
                    st.session_state.monitor_data = merge_close_centroids(raw_df, threshold=min_dist)
                    st.success("✅ Optimization complete!")
                    time.sleep(2)

    # --- STEP 6: REVIEW FINAL RESULTS ---
    if st.session_state.monitor_data is not None:
        st.markdown("#### Review Final Optimized Monitor Locations")
        final_df = st.session_state.monitor_data
        tab1, tab2 = st.tabs(["Optimized Monitors Map", "Download Data"])
        
        with tab1:
            map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
            m_final = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron", width=1700, height=700)
            folium.GeoJson(
                    st.session_state.boundary,
                    style_function=lambda x: {
                        'color': 'black',         # The color of the outline
                        'weight': 2,             # The thickness of the outline
                        'fillOpacity': 0.0,      # No fill (makes it transparent inside)
                    },
                    name='Airshed Boundary'
                ).add_to(m_final)
            for index, row in final_df.iterrows():
                folium.CircleMarker(location=[row['lat'], row['lon']], radius=8, color='#e63946', fill=True, fill_color='#e63946',
                                    popup=f"Monitor #{index+1}<br>Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}").add_to(m_final)
            st_folium(m_final, width=1700, height=700)
        with tab2:
            st.dataframe(final_df.style.format({'lat': '{:.5f}', 'lon': '{:.5f}'}))
            final_csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Monitor Locations CSV", data=final_csv, file_name="optimized_monitor_locations.csv", mime="text/csv")