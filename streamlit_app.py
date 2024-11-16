import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import folium
import time
from folium.features import GeoJsonPopup, GeoJsonTooltip
from streamlit_folium import st_folium
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from geometry import cleanup_names, stgeodataframe


@st.cache_data
def preprocess(geojson_file) -> pd.DataFrame:
    progress_bar = st.progress(0, text="Loading File")

    df = gpd.read_file(geojson_file)

    progress_bar.progress(10, text="Getting center")

    center_coords = [df.geometry.iloc[0].centroid.y,
                     df.geometry.iloc[0].centroid.x]

    progress_bar.progress(20, text="Generating Names")

    # Ensure all features have a 'name' property
    if 'name' not in df.columns:
        df['name'] = None  # Create the column if it doesn't exist

    progress_bar.progress(40, text="Generating Launch Points Names")

    # Launch Points: Filter for Points
    launch_points_df = df[df.geometry.type == "Point"].copy()
    # Fill missing 'name' values
    launch_points_df['name'] = launch_points_df.apply(
        lambda row: row['name'] if row['name'] else f"Launch Point {row.name}",
        axis=1
    )

    progress_bar.progress(60, text="Generating Scan Areas Names")

    # Scan Areas: Filter for Polygons (including MultiPolygons if needed)
    scan_areas_df = df[df.geometry.type == "Polygon"].copy()
    # Fill missing 'name' values
    scan_areas_df['name'] = scan_areas_df.apply(
        lambda row: row['name'] if row['name'] else f"Scan Area {row.name}",
        axis=1
    )

    progress_bar.progress(100, text="Finalizing")
    progress_bar.empty()

    return (center_coords, launch_points_df, scan_areas_df)


# @st.cache_data
# def split_polygon_into_squares(scan_areas_df, size):
#     bounds = polygon.bounds
#     minx, miny, maxx, maxy = bounds

#     # Create a local projection centered on the polygon
#     local_utm = pyproj.Proj(proj='utm', zone=int(
#         (minx + maxx) / 2 // 6) + 31, ellps='WGS84')
#     wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
#     project_to_utm = pyproj.Transformer.from_proj(wgs84, local_utm).transform
#     project_to_wgs84 = pyproj.Transformer.from_proj(local_utm, wgs84).transform

#     # Convert the polygon to the local UTM coordinates
#     utm_polygon = transform(project_to_utm, polygon)

#     # Get the bounds in the local UTM coordinates
#     minx, miny, maxx, maxy = utm_polygon.bounds

#     squares = []
#     x_coords = np.arange(minx, maxx, size)
#     y_coords = np.arange(miny, maxy, size)

#     for x in x_coords:
#         for y in y_coords:
#             square = Polygon([
#                 (x, y),
#                 (x + size, y),
#                 (x + size, y + size),
#                 (x, y + size),
#                 (x, y)
#             ])
#             intersection = utm_polygon.intersection(square)
#             if not intersection.is_empty:
#                 squares.append(transform(project_to_wgs84, intersection))

#     return squares

@st.cache_data
def process(launch_points_df, scan_areas_df) -> pd.DataFrame:
    progress_bar = st.progress(0, text="Computing projections")

    # Create a local projection centered on the polygon
    minx, miny, maxx, maxy = scan_areas_df.total_bounds
    local_utm = pyproj.Proj(proj='utm', zone=int(
        (minx + maxx) / 2 // 6) + 31, ellps='WGS84')
    wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
    project_to_utm = pyproj.Transformer.from_proj(wgs84, local_utm).transform
    project_to_wgs84 = pyproj.Transformer.from_proj(local_utm, wgs84).transform

    # Apply the transformation to launch points
    launch_points = launch_points_df.copy()
    launch_points["geometry"] = launch_points["geometry"].apply(
        lambda geom: transform(project_to_utm, geom))
    launch_points.set_crs(local_utm.srs, inplace=True)

    # Apply the transformation to scan areas
    scan_areas = scan_areas_df.copy()
    scan_areas["geometry"] = scan_areas["geometry"].apply(
        lambda geom: transform(project_to_utm, geom))
    scan_areas.set_crs(local_utm.srs, inplace=True)

    progress_bar.progress(1, text="Slicing scan areas")






    

    progress_bar.progress(100, text="Finalizing")
    time.sleep(1.0)
    progress_bar.empty()

    return ([])


###############################################################################
st.markdown("# 📶 LTE Survey Mission Generator")

###############################################################################
st.markdown("## 1. Upload input file")

geojson_file = st.file_uploader(
    "Upload a geojson file (e.g. from geojson.io) containing polygons covering the area you want to scan and points representing launch/land locations:",
    type=["json", "geojson"])

if geojson_file is None:
    st.stop()

(center_coords, launch_points_df, scan_areas_df) = preprocess(geojson_file)

with st.expander("View map of scan areas and launch points"):
    m = folium.Map(location=center_coords, zoom_start=12)
    folium.GeoJson(
        pd.concat([launch_points_df, scan_areas_df]),
        popup=GeoJsonPopup(
            fields=["name"],
            aliases=["Name:"],
            localize=True,
            labels=True,
            style="background-color: yellow;",
        ),
    ).add_to(m)
    st_folium(m, width=700, height=500, return_on_hover=False)

###############################################################################
st.markdown("## 2. Customize parameters")

name_template = st.text_input(
    "Mission name template:", value="LTE Scan - {date} - {index}")

with st.expander("Mission Planning Parameters"):

    st.markdown(
        """
        We generate a lawnmower pattern to scan the area with a given spacing between passes.
        We slice the scan area into separate corridors of specified width, along the first pass axis.
        For each corridor, we then slice it again into separate slices across the first pass axis. 
        We compute the size of each slice such that each mission is as long as possible, but no more than target duration.
        We then fly the drone at specific speed and altitude along these passes.
        """)

    passes = st.multiselect(
        "Which scan passes do you want to fly?",
        ["North-South", "East-West"],
        ["North-South"],
    )

    mission_duration = st.number_input(
        "Target mission duration, in minutes:", min_value=0, value=20)
    st.markdown(
        f"Mission duration = {mission_duration*60:.0f}s = {mission_duration:.0f}min{(mission_duration*60)%60:.0f}s")

    corridor_width = st.number_input(
        "Width of mission corridors, in meters:", min_value=100, value=800)
    st.markdown(
        f"Width = {corridor_width:.0f}m = {corridor_width*3.28084:.0f}ft = {corridor_width*1.09361:.0f}yd = {corridor_width/1609:.2f} Miles = {corridor_width/1852:.2f} Nautical Miles")

    spacing = st.number_input(
        "Spacing between passes in meters:", min_value=10, value=100)
    st.markdown(
        f"Spacing = {spacing:.0f}m = {spacing*3.28084:.0f}ft = {spacing*1.09361:.0f}yd = {spacing/1609:.2f} Miles = {spacing/1852:.2f} Nautical Miles")

    speed = st.number_input(
        "Flight speed in meters per second:", min_value=0, value=16)
    st.markdown(
        f"Flight speed = {speed:.0f}m/s = {speed*3.6:.1f}km/h = {speed*3600/1609:.1f}Mph = {speed*3600/1852:.1f}kts")

    altitude = st.number_input(
        "Flight altitude in meters:", min_value=10, value=61)
    st.markdown(
        f"Altitude = {altitude:.0f}m = {altitude*3.28084:.0f}ft = {altitude*1.09361:.0f}yd = {altitude/1609:.2f} Miles = {altitude/1852:.2f} Nautical Miles")

with st.expander("Return Settings"):

    rtx_height = st.number_input(
        "Return Height in meters:", min_value=10, value=61)
    st.markdown(
        f"Return Height = {rtx_height:.0f}m = {rtx_height*3.28084:.0f}ft = {rtx_height*1.09361:.0f}yd = {rtx_height/1609:.2f} Miles = {rtx_height/1852:.2f} Nautical Miles")

    rtx_speed = st.number_input(
        "Return Speed in meters per second:", min_value=0, value=16)
    st.markdown(
        f"Return Speed = {rtx_speed:.0f}m/s = {rtx_speed*3.6:.1f}km/h = {rtx_speed*3600/1609:.1f}Mph = {rtx_speed*3600/1852:.1f}kts")

    rtx_wait = st.number_input(
        "Wait Before Return on Lost Connection:", min_value=0, value=30)
    st.markdown(
        f"Wait Before Return on Lost Connection = {rtx_wait:.0f}s = {rtx_wait/60:.0f}min{rtx_wait%60:.0f}s")

with st.expander("Cost Estimation Parameters"):
    cost_fixed = st.number_input(
        "Fixed base cost for the whole program:", min_value=0, value=5000)

    cost_per_flight = st.number_input(
        "Fixed cost per flight mission:", min_value=0, value=100)

    cost_per_flight_hour = st.number_input(
        "Variable cost per flight hour:", min_value=0, value=50)


###############################################################################
st.markdown("## 3. Compute missions")

if st.button("Compute Missions"):
    (missions) = process(launch_points_df, scan_areas_df)
else:
    st.stop()

with st.expander("View map of missions"):
    m = folium.Map(location=center_coords, zoom_start=12)
    folium.GeoJson(
        pd.concat([launch_points_df, scan_areas_df]),
        popup=GeoJsonPopup(
            fields=["name"],
            aliases=["Name:"],
            localize=True,
            labels=True,
            style="background-color: yellow;",
        ),
    ).add_to(m)
    st_folium(m, width=700, height=500, return_on_hover=False)


###############################################################################
st.markdown("## 4. Upload Missions to Cloud")

if st.button("Upload Missions"):
    # asyncio.run(upload_missions())
    st.text("OK")
