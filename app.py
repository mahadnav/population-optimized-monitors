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
import json
import os
import time
from geopy.distance import geodesic
from helpers.utils import classify_population_density, randomize_initial_cluster, weighted_kmeans

if "population_grid" not in st.session_state:
    st.session_state["population_grid"] = None
if "population_computed" not in st.session_state:
    st.session_state["population_computed"] = False
if "monitor_data" not in st.session_state:
    st.session_state["monitor_data"] = None

def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def merge_close_centroids(centroids, threshold=2):
    merged_centroids = []
    used = set()
    
    for i, row1 in centroids.iterrows():
        if i in used:
            continue
        close_centroids = [row1]
        for j, row2 in centroids.iterrows():
            if i != j and j not in used:
                distance = calculate_distance((row1['lat'], row1['lon']), 
                                              (row2['lat'], row2['lon']))
                if distance < threshold:
                    close_centroids.append(row2)
                    used.add(j)
        if len(close_centroids) > 1:
            mean_lat = np.mean([c['lat'] for c in close_centroids])
            mean_long = np.mean([c['lon'] for c in close_centroids])
            merged_centroids.append({'lat': mean_lat, 
                                     'lon': mean_long})
        else:
            merged_centroids.append({'lat': row1['lat'], 
                                     'lon': row1['lon']})
        used.add(i)

    new_centroids = pd.DataFrame(merged_centroids)
    
    # Check if any centroids are within the threshold distance in the new centroids dataframe
    for i, row1 in new_centroids.iterrows():
        for j, row2 in new_centroids.iterrows():
            if i != j:
                distance = calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon']))
                if distance < threshold:
                    return merge_close_centroids(new_centroids, threshold)
    
    return new_centroids


st.set_page_config(page_title="Population-Centric Monitoring Network", layout="wide")

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

col1, col2 = st.columns([1, 5], vertical_alignment="center")
with col1:
    # Make sure 'logo.png' is the correct path to your logo file
    st.image("logo.jpeg", width=150)
with col2:
    st.title("Population-Centric Air Quality Monitor Optimization")
    st.markdown("### A tool to strategically place air quality monitors based on population density.")
    st.write("Developed by Mahad Naveed and the PakAirQuality (PAQI) Team.")

st.divider() 

st.markdown("### Define Your Airshed")

# --- Initial Map Display ---
m = folium.Map(zoom_start=8)
from folium.plugins import Draw
Draw(export=False, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 
                                 'circlemarker': False, 'marker': False, 'polyline': False}).add_to(m)
st_map = st_folium(m, width=1700, height=700, returned_objects=["last_active_drawing"])

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
        st.success(f"Grid generated with {len(gdf)} cells.")

        tif_file = get_worldpop_data()

        # --- Population Calculation Only If Not Already Done ---
        if not st.session_state["population_computed"]:
            with st.spinner("Analyzing population data..."): 
                total_geometries = len(gdf)
                chunk_size = 10
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

        st.success("✅ Population values computed!")

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
                    'weight': 0,
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

        fig = sns.displot(
        data=density_df, 
        x='population', 
        hue='Density',
        palette='inferno', 
        edgecolor='k', 
        linewidth=0.5, 
        bins=45,
        kind='hist',
        kde=True)
        fig.figure.set_size_inches(8, 3)
        fig.set_axis_labels("Population Density", "Frequency")
        st.pyplot(fig)


        st.subheader("Cluster Analysis with Weighted K-Means")

        # --- Step 1: Place User Inputs at the top. They will always be visible. ---
        col1, col2 = st.columns(2)
        with col1:
            low_monitors = st.number_input(
                "Number of Monitors for Low Density", 
                min_value=2, max_value=100, value=11, key="low_clusters"
            )
        with col2:
            high_monitors = st.number_input(
                "Number of Monitors for High Density", 
                min_value=1, max_value=100, value=15, key="high_clusters"
            )

        # --- Step 2: Add a button to trigger the expensive calculation. ---
        run_button = st.button("🚀 Run Monitor Optimization Analysis")

        # --- Step 3: Run the calculation ONLY when the button is clicked. ---
        if run_button:
            with st.spinner("Optimizing monitor locations..."):
                # Get the data needed for the calculation
                vals = density_df[['population', 'long', 'lat', 'Density']].copy()
                low = vals[vals['Density'] == 'Low'][['population', 'long', 'lat']]
                high = vals[vals['Density'] == 'High'][['population', 'long', 'lat']]

                # --- Low & High density calculations ---
                # (This logic is correct, keeping it collapsed for brevity)
                sampled_low = low.sample(int(0.7 * len(low)))
                centers_low = randomize_initial_cluster(sampled_low, low_monitors)
                _, centers_low, _, _ = weighted_kmeans(low, centers_low, low_monitors)
                low_centroids = pd.DataFrame(centers_low)
                low_clat = [x[0][1] for _, x in low_centroids.iterrows()]
                low_clong = [x[0][0] for _, x in low_centroids.iterrows()]
                
                sampled_high = high.sample(int(0.7 * len(high)))
                centers_high = randomize_initial_cluster(sampled_high, high_monitors)
                _, centers_high, _, _ = weighted_kmeans(high, centers_high, high_monitors)
                high_centroids = pd.DataFrame(centers_high)
                high_clat = [x[0][1] for _, x in high_centroids.iterrows()]
                high_clong = [x[0][0] for _, x in high_centroids.iterrows()]

                # --- Combine and merge the results ---
                low_df = pd.DataFrame({'lat': low_clat, 'lon': low_clong})
                high_df = pd.DataFrame({'lat': high_clat, 'lon': high_clong})
                raw_df = pd.concat([low_df, high_df], ignore_index=True)
                final_monitors_df = merge_close_centroids(raw_df, threshold=2) 
                
                # Save the final result to session state
                st.session_state["monitor_data"] = final_monitors_df
                st.success("✅ Analysis complete! Results are now stored in the session.")
            
            colors = [
                '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a',
                '#ffff99',
                '#b15928',
                '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a',
                '#ffff99',
                '#b15928',
                '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a',
                '#ffff99',
                '#b15928',
                '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a',
                '#ffff99',
                '#b15928',
                '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a',
                '#ffff99',
                '#b15928',
                '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a',
                '#ffff99',
                '#b15928']


        if st.session_state["monitor_data"] is not None:
            st.subheader("Final Optimized Monitor Locations")
            
            # Retrieve the data from the session
            final_monitors_df = st.session_state["monitor_data"]

            # Create and display the map
            map_center = [final_monitors_df['lat'].mean(), final_monitors_df['lon'].mean()]
            m = folium.Map(location=map_center, zoom_start=11)

            for index, row in final_monitors_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    color='#FF0000',
                    fill=True,
                    fill_color='#FF0000',
                    fill_opacity=0.6,
                    popup=f"Point {index+1}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}"
                ).add_to(m)

            st_folium(m, width=1700, height=700)

            # Display the download button
            st.download_button(
                "Click to download monitor locations",
                data=final_monitors_df.to_csv(index=False).encode('utf-8'),
                file_name="optimized_monitor_locations.csv",
                mime="text/csv"
            )

    

    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
