import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import box
from rasterstats import zonal_stats
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import json
import os
from helpers.utils import classify_population_density, randomize_initial_cluster, weighted_kmeans

if "drawing" not in st.session_state:
    st.session_state.drawing = None
if "population_grid" not in st.session_state:
    st.session_state.population_grid = None
if "population_computed" not in st.session_state:
    st.session_state.population_computed = False

# --- KEY CHANGE 2: Cache the creation of BOTH maps ---
# This is your original cached function for the results map. IT IS CORRECT.
@st.cache_data
def create_results_map(dataframe):
    st.write("--- Creating Results Map (Cached) ---")
    map_center = [dataframe['lat'].mean(), dataframe['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=11)
    for index, row in dataframe.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8, color='#FF0000', fill=True, fill_color='#FF0000', fill_opacity=0.6,
            popup=f"Point {index+1}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}"
        ).add_to(m)
    return m

# This is a NEW cached function for the initial drawing map.
# Using @st.cache_resource is often better for complex objects like maps with plugins.
@st.cache_resource
def create_drawing_map():
    st.write("--- Creating Drawing Map (Cached) ---")
    m = folium.Map(location=[30.1575, 71.5249], zoom_start=10) # Centered on Multan
    Draw(export=False, draw_options={'rectangle': True, 'polygon': False, 'circle': False, 'marker': False, 'polyline': False}).add_to(m)
    return m

@st.cache_data(hash_funcs={gpd.GeoDataFrame: id})
def create_population_grid_map(gdf, map_bounds):
    """
    Creates and caches the Folium choropleth map of the population grid.
    This is slow and should only run once.
    """
    st.write("--- Creating Population Grid Map (This should only print once!) ---")

    # Unpack map bounds
    min_lat, max_lat, min_lon, max_lon = map_bounds

    # Create the base map
    m_grid = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=8)
    
    # --- All the GeoJson and Styling logic now lives inside the cached function ---
    gdf_copy = gdf.copy().fillna(0).reset_index(drop=True)
    gdf_copy['str_id'] = gdf_copy['id'].astype(str)
    geojson_data = json.loads(gdf_copy.to_json())

    for i, feature in enumerate(geojson_data['features']):
        feature['id'] = gdf_copy.loc[i, 'str_id']

    pop_min = gdf_copy['population'].min()
    pop_max = gdf_copy['population'].max()
    colormap = cm.get_cmap('plasma')
    norm = colors.Normalize(vmin=pop_min, vmax=pop_max)

    def style_function(feature):
        # We need to look up the population from the gdf_copy
        pop_series = gdf_copy.loc[gdf_copy['str_id'] == feature['id'], 'population']
        if pop_series.empty:
            return {'fillOpacity': 0, 'weight': 0}
        
        pop = pop_series.values[0]

        if pop == 0 or np.isnan(pop):
            return {'fillOpacity': 0, 'weight': 0}
        else:
            color = colors.rgb2hex(colormap(norm(pop))[:3])
            return {
                'fillColor': color, 'color': 'black',
                'weight': 0.5, 'fillOpacity': 0.7,
            }

    folium.GeoJson(
        geojson_data,
        name='Population Grid',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['population'], aliases=['Population:'])
    ).add_to(m_grid)

    folium.LayerControl().add_to(m_grid)

    return m_grid








st.set_page_config(page_title="Grid Generator for Airshed", layout="wide")
st.title("📍 Define Airshed and Generate Population Grid with WorldPop")

st.markdown("""
Draw a rectangle on the map to define your airshed boundary.
""")

# Get the cached drawing map
drawing_map = create_drawing_map()
# Display it and capture the output
st_map_output = st_folium(drawing_map, width=1500, height=500, returned_objects=["last_active_drawing"])

# --- KEY CHANGE 3: Save the drawing to Session State ---
# If the user just finished a drawing, save it.
if st_map_output and st_map_output.get("last_active_drawing"):
    st.session_state.drawing = st_map_output["last_active_drawing"]

def get_worldpop_data():
    """Handles the upload of the WorldPop GeoTIFF file."""
    uploaded_file = st.file_uploader("📂 Upload a WorldPop GeoTIFF (.tif) file", type=["tif", "tiff"])
    if not uploaded_file:
        st.warning("Please upload a raster file to continue.")
        st.stop()
    return uploaded_file

if st.session_state.drawing:
    st.info("Airshed boundary captured! You can now proceed.")
    geom = st.session_state.drawing
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

        st.subheader("🗺️ Grid with Population")

        # Pass the necessary data to the cached function
        map_bounds = (min_lat, max_lat, min_lon, max_lon)
        population_map = create_population_grid_map(gdf, map_bounds)

        # Display the cached map. This is now fast and won't cause a loop.
        st_folium(population_map, width=1500, height=500)

        csv = gdf.to_csv(index=False).encode('utf-8')

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

        fig = sns.displot(
        data=density_df, 
        x='population', 
        hue='Density',
        palette='inferno', 
        edgecolor='k', 
        linewidth=0.5, 
        bins=45,
        kind='hist')
        fig.figure.set_size_inches(8, 3)
        fig.set_axis_labels("Population Density", "Frequency")
        st.pyplot(fig)


        st.subheader("Cluster Analysis with Weighted K-Means")

        vals = density_df[['population', 'long', 'lat']].copy()
        sampled = vals.sample(int(0.7 * len(density_df)))
        centers = randomize_initial_cluster(sampled, 11)
        points, centers, iters, sse = weighted_kmeans(vals, centers, 11)

        # Compute total population per cluster
        cluster_populations = points.groupby('cluster')['population'].sum().to_dict()

        # Extract centroids
        centroids = pd.DataFrame(centers)
        clat = [x[0][1] for _, x in centroids.iterrows()]
        clong = [x[0][0] for _, x in centroids.iterrows()]

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


        centroids_df = pd.DataFrame({
            'lat': clat,
            'lon': clong
        })

        results_folium_map = create_results_map(centroids_df)
        st.subheader("Optimized Monitor Locations")
        st_folium(results_folium_map, width=1200, height=600)

    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
