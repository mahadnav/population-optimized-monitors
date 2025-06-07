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
import seaborn as sns
import tempfile
import json
import os
from geopy.distance import geodesic
from helpers.utils import classify_population_density, randomize_initial_cluster, weighted_kmeans

# --- Page Configuration (do this first!) ---
st.set_page_config(
    page_title="PAQI - AQM Network Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection ---
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# --- Session State Initialization ---
# This ensures variables persist across reruns
if "boundary" not in st.session_state:
    st.session_state.boundary = None
if "population_grid" not in st.session_state:
    st.session_state.population_grid = None
if "monitor_data" not in st.session_state:
    st.session_state.monitor_data = None

# --- Helper Functions (Your existing functions) ---
# (Keeping these collapsed for brevity, no changes were needed)
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def merge_close_centroids(centroids, threshold=2):
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
    for i, row1 in new_centroids.iterrows():
        for j, row2 in new_centroids.iterrows():
            if i != j:
                distance = calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon']))
                if distance < threshold:
                    return merge_close_centroids(new_centroids, threshold)
    return new_centroids

# --- Main App UI ---

# --- HEADER SECTION ---
col1, col2 = st.columns([1, 5], vertical_alignment="center")

with col1:
    # Make sure 'logo.png' is the correct path to your logo file
    st.image("logo.jpeg", width=150)

with col2:
    st.title("Population-Centric Air Quality Monitor Optimization")
    st.markdown("### A tool to strategically place air quality monitors based on population density.")
    st.write("Developed by Mahad Naveed and the PakAirQuality (PAQI) Team.")

st.divider()

# --- STEP 1: DEFINE AIRSHED BOUNDARY ---
st.header("Step 1: Define Your Airshed", anchor=False)
st.markdown("Use the drawing tool on the map to draw a rectangle over your area of interest. The analysis will begin automatically once a rectangle is drawn.")

map_center = [25, 65] # Centered on Pakistan
m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")
draw_plugin = folium.plugins.Draw(
    export=False,
    draw_options={'rectangle': True, 'polygon': False, 'circlemarker': False, 'polyline': False}
)
draw_plugin.add_to(m)

st_map = st_folium(m, width=1700, height=500, returned_objects=["last_active_drawing"])

# Check if a drawing has been made and update the session state
if st_map and st_map.get("last_active_drawing"):
    st.session_state.boundary = st_map["last_active_drawing"]

# --- STEP 2: ANALYZE POPULATION DATA ---
if st.session_state.boundary:
    st.markdown("Step 2: Analyze Population Data", anchor=False)
    
    # Using a container to group this step's logic and UI
    with st.container(border=True):
        geom = st.session_state.boundary
        if not (geom and geom.get("geometry") and geom["geometry"]["type"] == "Polygon"):
            st.warning("Please draw a valid rectangle on the map.")
            st.stop()

        coords = geom["geometry"]["coordinates"][0]
        lons, lats = zip(*coords)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        st.info(f"Airshed boundary defined from Lat: {min_lat:.4f} to {max_lat:.4f}, Lon: {min_lon:.4f} to {max_lon:.4f}")

        # --- File Uploader ---
        tif_file = st.file_uploader(
            "📂 Upload the WorldPop GeoTIFF (.tif) file for this region",
            type=["tif", "tiff"]
        )

        if tif_file:
            if st.session_state.get("population_grid") is None:
                with st.spinner("Analyzing population data. This may take a moment..."):
                    # Grid Generation
                    resolution = 0.01
                    lat_points, lon_points = np.arange(min_lat, max_lat, resolution), np.arange(min_lon, max_lon, resolution)
                    records = [
                        {"id": i * len(lon_points) + j + 1, "geometry": box(lon, lat, lon + resolution, lat + resolution)}
                        for i, lat in enumerate(lat_points) for j, lon in enumerate(lon_points)
                    ]
                    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
                    
                    # Population Calculation using a temporary file to save memory
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                        # Write the uploaded file's bytes to the temporary file
                        tmp.write(tif_file.getvalue())
                        tmp_path = tmp.name # Get the path to the temporary file

                    # Now, run zonal_stats using the file path instead of the in-memory object
                    stats = zonal_stats(gdf, tmp_path, stats="sum", all_touched=True)
                    
                    # Clean up the temporary file from the disk
                    os.remove(tmp_path) 
                    
                    gdf["population"] = [s['sum'] if s and s['sum'] is not None else 0 for s in stats]
                    st.session_state.population_grid = gdf[gdf['population'] > 0].reset_index(drop=True)
                    
            st.success(f"✅ Population data processed for **{len(gdf)}** grid cells.")
            
            # --- Display Population Map and Data in Tabs ---
            tab1, tab2, tab3 = st.tabs(["🗺️ Population Map", "📊 Population Distribution", "📥 Download Data"])

            with tab1:
                st.subheader("Population Heatmap")
                m_grid = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=8, tiles="CartoDB positron")
                
                # Create and add colormap to map
                pop_min, pop_max = gdf['population'].min(), gdf['population'].max()
                colormap = cm.get_cmap('viridis')
                scalar_map = folium.colormap.LinearColormap(
                    colors=[colors.rgb2hex(colormap(i)) for i in range(colormap.N)],
                    vmin=pop_min, vmax=pop_max, caption="Population Count"
                )
                m_grid.add_child(scalar_map)

                folium.GeoJson(
                    gdf,
                    style_function=lambda feature: {
                        'fillColor': colormap(feature['properties']['population'] / pop_max) if pop_max > 0 else 'transparent',
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.7 if feature['properties']['population'] > 0 else 0,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['population'], aliases=['Population:'])
                ).add_to(m_grid)
                st_folium(m_grid, width=1500, height=500, returned_objects=[])

            with tab2:
                st.subheader("Population Density Analysis")
                density_df = classify_population_density(gdf.copy())
                fig = sns.displot(data=density_df, x='population', hue='Density', palette='viridis', kind='hist', kde=True)
                fig.set_axis_labels("Population Count per Cell", "Number of Cells")
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Download Processed Population Data")
                st.markdown("Download the processed grid data with population counts as a CSV file.")
                grid_csv = gdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Population Grid CSV",
                    data=grid_csv,
                    file_name="population_grid.csv",
                    mime="text/csv"
                )


# --- STEP 3: OPTIMIZE MONITOR LOCATIONS ---
if st.session_state.get("population_grid") is not None:
    st.header("Step 3: Optimize Monitor Locations", anchor=False)
    
    with st.form("optimization_form"):
        st.markdown("Set the number of monitors to place in low and high population density areas, then run the analysis.")
        
        col1, col2 = st.columns(2)
        with col1:
            low_monitors = st.number_input("Monitors for Low Density Areas", min_value=1, value=5)
        with col2:
            high_monitors = st.number_input("Monitors for High Density Areas", min_value=1, value=10)

        submitted = st.form_submit_button("🚀 Run Optimization", use_container_width=True)

        if submitted:
            with st.spinner("Running weighted K-Means clustering to find optimal locations..."):
                gdf = st.session_state.population_grid
                density_df = classify_population_density(gdf.copy())
                density_df['long'] = density_df.geometry.centroid.x
                density_df['lat'] = density_df.geometry.centroid.y
                
                # Run clustering logic
                vals = density_df[['population', 'long', 'lat', 'Density']].copy()
                low = vals[vals['Density'] == 'Low']
                high = vals[vals['Density'] == 'High']
                
                if not low.empty:
                    _, centers_low, _, _ = weighted_kmeans(low, randomize_initial_cluster(low, low_monitors), low_monitors)
                    low_centroids = pd.DataFrame(centers_low, columns=['coords'])
                    low_df = pd.DataFrame({'lat': [c[1] for c in low_centroids['coords']], 'lon': [c[0] for c in low_centroids['coords']]})
                else:
                    low_df = pd.DataFrame(columns=['lat', 'lon'])

                if not high.empty:
                    _, centers_high, _, _ = weighted_kmeans(high, randomize_initial_cluster(high, high_monitors), high_monitors)
                    high_centroids = pd.DataFrame(centers_high, columns=['coords'])
                    high_df = pd.DataFrame({'lat': [c[1] for c in high_centroids['coords']], 'lon': [c[0] for c in high_centroids['coords']]})
                else:
                    high_df = pd.DataFrame(columns=['lat', 'lon'])
                
                raw_df = pd.concat([low_df, high_df], ignore_index=True)
                final_monitors_df = merge_close_centroids(raw_df, threshold=2)
                st.session_state.monitor_data = final_monitors_df
    
    # --- Display Final Results (outside the form) ---
    if st.session_state.get("monitor_data") is not None:
        final_df = st.session_state.monitor_data
        st.success(f"✅ Optimization complete! Found **{len(final_df)}** final monitor locations.")
        
        tab1, tab2 = st.tabs(["🗺️ Final Monitor Map", "📥 Download Locations"])

        with tab1:
            st.subheader("Optimized Monitor Placement")
            map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
            m_final = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron")
            
            # Add population heatmap as a base layer
            folium.GeoJson(
                st.session_state.population_grid,
                style_function=lambda feature: {'fillColor': 'grey', 'color': 'transparent', 'fillOpacity': 0.2},
            ).add_to(m_final)

            # Add monitor locations
            for index, row in final_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8, color='#e63946', fill=True, fill_color='#e63946', fill_opacity=0.9,
                    popup=f"Monitor #{index+1}<br>Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}"
                ).add_to(m_final)
            st_folium(m_final, width=1700, height=700, returned_objects=[])

        with tab2:
            st.subheader("Download Final Monitor Locations")
            st.dataframe(final_df.style.format({'lat': '{:.5f}', 'lon': '{:.5f}'}))
            final_csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Monitor Locations CSV",
                data=final_csv,
                file_name="optimized_monitor_locations.csv",
                mime="text/csv"
            )