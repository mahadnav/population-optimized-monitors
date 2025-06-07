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
import pydeck as pdk
import json
import os
from helpers.utils import classify_population_density, randomize_initial_cluster, weighted_kmeans

if "population_grid" not in st.session_state:
    st.session_state["population_grid"] = None
if "population_computed" not in st.session_state:
    st.session_state["population_computed"] = False
if "monitor_data" not in st.session_state:
    st.session_state["monitor_data"] = None


st.set_page_config(page_title="Grid Generator for Airshed", layout="wide")
st.title("📍 Define Airshed and Generate Population Grid with WorldPop")

st.markdown("""
Draw a rectangle on the map to define your airshed boundary.
""")

# --- Initial Map Display ---
m = folium.Map(zoom_start=8)
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

        if st.session_state["monitor_data"] is None:
            st.info("First run: Optimizing monitor locations...")
            centroids_df = pd.DataFrame({
                'lat': clat,
                'lon': clong
            })
            st.success("✅ Monitor locations optimized and saved to session.")
            st.session_state["monitor_data"] = centroids_df
        else:
            st.info("Retrieving monitor locations from session state...")
            centroids_df = st.session_state["monitor_data"]
            st.success("✅ Monitor locations retrieved from session.")



        map_center = [centroids_df['lat'].mean(), centroids_df['lon'].mean()]

        # The `zoom_start` parameter controls the initial zoom level.
        m = folium.Map(location=map_center, zoom_start=11)


        # --- Step 4: Add Points to the Map ---
        # We will loop through each row in our DataFrame.
        for index, row in centroids_df.iterrows():
            # For each point, add a CircleMarker.
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,  # The size of the circle marker
                color='#FF0000',  # The color of the circle's border (red)
                fill=True,
                fill_color='#FF0000',  # The color inside the circle
                fill_opacity=0.6,
                popup=f"Point {index+1}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}" # What shows up when you click
            ).add_to(m)


        # --- Step 5: Display the Map in Streamlit ---
        # Use st_folium to render the Folium map object.
        st.subheader("Interactive Folium Map")
        st_data = st_folium(m, width=1200, height=600)

    

    else:
        st.warning("Please draw a rectangle to define the airshed.")
else:
    st.info("Use the map to draw a rectangle for your airshed boundary.")
