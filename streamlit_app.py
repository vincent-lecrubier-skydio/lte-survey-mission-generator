from typing import Union
from shapely import LineString
from geometry import generate_oriented_slices, compute_total_mission_path, generate_passes, project_df, cleanup_names, generate_corridors, stgeodataframe
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
import colorsys
import math


def generate_random_saturated_color(index):
    # Generate a random hue (0-360 degrees)
    hue = (index * 137) % 360  # A multiplier like 137 ensures a spread of hues
    # Saturation and lightness are set for high saturation and medium brightness
    saturation = 100  # Fully saturated
    lightness = 50    # Medium lightness for vivid color
    r, g, b = colorsys.hls_to_rgb(
        hue / 360.0, lightness / 100.0, saturation / 100.0)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def compute_mission_duration(
    mission_path: LineString,
    altitude: float,
    horizontal_speed_meters_per_second: float,
    time_lost_per_waypoint_seconds=6.0,
    ascent_speed_meters_per_second=8.0,
    descent_speed_meters_per_second=6.0,
) -> Union[float, None]:
    """
    Compute the duration of a mission given the speed and altitude.
    """
    if mission_path is None:
        return None
    duration_travel = mission_path.length / horizontal_speed_meters_per_second
    duration_ascent = altitude / ascent_speed_meters_per_second
    duration_descent = altitude / descent_speed_meters_per_second
    duration_waypoints = len(mission_path.coords) * \
        time_lost_per_waypoint_seconds
    return duration_travel + duration_ascent + duration_descent + duration_waypoints


@st.cache_data
def preprocess(geojson_file) -> pd.DataFrame:
    preprocess_progress_bar = st.progress(0, text="Loading File")

    df = gpd.read_file(geojson_file)

    preprocess_progress_bar.progress(10, text="Getting center")

    center_coords = [df.geometry.iloc[0].centroid.y,
                     df.geometry.iloc[0].centroid.x]

    preprocess_progress_bar.progress(20, text="Generating Names")

    # Ensure all features have a 'name' property
    if 'name' not in df.columns:
        df['name'] = None  # Create the column if it doesn't exist

    preprocess_progress_bar.progress(40, text="Generating Launch Points Names")

    # Launch Points: Filter for Points
    launch_points_df = df[df.geometry.type == "Point"].copy()
    # Fill missing 'name' values
    launch_points_df['name'] = launch_points_df.apply(
        lambda row: row['name'] if row['name'] else f"Launch Point {row.name}",
        axis=1
    )

    preprocess_progress_bar.progress(60, text="Generating Scan Areas Names")

    # Scan Areas: Filter for Polygons (including MultiPolygons if needed)
    scan_areas_df = df[df.geometry.type == "Polygon"].copy()
    # Fill missing 'name' values
    scan_areas_df['name'] = scan_areas_df.apply(
        lambda row: row['name'] if row['name'] else f"Scan Area {row.name}",
        axis=1
    )

    preprocess_progress_bar.progress(100, text="Finalizing")
    preprocess_progress_bar.empty()

    return (center_coords, launch_points_df, scan_areas_df)

# @st.cache_data


def process(
        geojson_file, _launch_points_df, _scan_areas_df,
        corridor_direction, corridor_width, pass_direction, pass_spacing, crosshatch,
        altitude, speed, max_mission_duration
) -> pd.DataFrame:
    process_progress_bar = st.progress(0, text="Computing projections")

    # Create a local projection centered on the polygon
    minx, miny, maxx, maxy = scan_areas_df.total_bounds
    local_utm = pyproj.Proj(proj='utm', zone=int(
        (minx + maxx) / 2 // 6) + 31, ellps='WGS84')
    wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
    project_to_utm = pyproj.Transformer.from_proj(wgs84, local_utm)
    project_to_wgs84 = pyproj.Transformer.from_proj(local_utm, wgs84)

    # # Apply the transformation to launch points
    launch_points = project_df(project_to_utm, _launch_points_df)

    # Apply the transformation to scan areas
    scan_areas = project_df(project_to_utm, _scan_areas_df)

    process_progress_bar.progress(10, text="Generating slices")

    iteration_spacing = pass_spacing
    slices = generate_oriented_slices(
        scan_areas, corridor_direction, corridor_width, iteration_spacing)

    process_progress_bar.progress(20, text="Generating passes")

    passes = generate_passes(
        scan_areas, pass_direction, pass_spacing)
    if crosshatch:
        # Add crosshatch perpendicular passes
        passes_crosshatch = generate_passes(
            scan_areas, pass_direction + 90, pass_spacing)
    else:
        passes_crosshatch = None

    process_progress_bar.progress(50, text="Generating optimal missions")

    start_slice = 0
    end_slice = slices.shape[0]

    missions_optim = []
    current_start = start_slice

    while current_start < end_slice:

        process_progress_bar.progress(50+50*math.floor((current_start-start_slice)/(
            end_slice-start_slice)), text="Generating optimal missions")

        # Binary search to find the best range
        low, high = current_start + 1, end_slice
        best_end = None
        best_reward = float('-inf')

        while low <= high:
            mid = (low + high) // 2

            mission_path = compute_total_mission_path(
                launch_points, slices, passes, passes_crosshatch, current_start, mid)
            mission_duration = compute_mission_duration(
                mission_path, altitude, speed)
            reward = mid-current_start
            st.text(f"{current_start}, {mid}: {mission_duration}")

            if mission_duration is None:  # Invalid range, increase size
                low = mid + 1
            elif mission_duration > max_mission_duration:  # Range too large, decrease size
                high = mid - 1
            else:  # Valid range, optimize for maximum reward
                if reward > best_reward:
                    best_reward = reward
                    best_end = mid
                low = mid + 1  # Explore larger ranges

        # If no valid range is found, terminate the loop
        if best_end is None:
            raise ValueError(
                "No mission found for the given constraints, try making target duration larger")

        # Finalize the missions
        missions_optim.append((current_start, best_end,
                               mission_path, mission_duration))
        current_start = best_end  # Update start for the next range

    # missions_optim = [
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 0, 20),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 20, 37),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 37, 92),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 92, 165),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 165, 200),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 200, 275),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 275, 342),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 342, 384),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 384, 427),
    #     compute_total_mission_path(
    #         launch_points, slices, passes, passes_crosshatch, 427, 560),
    # ]

    missions = gpd.GeoDataFrame({
        'geometry': [mission_path for (start, end, mission_path, mission_duration) in missions],
        'start': [start for (start, end, mission_path, mission_duration) in missions],
        'end': [end for (start, end, mission_path, mission_duration) in missions],
        'duration': [mission_duration for (start, end, mission_path, mission_duration) in missions],
        'color': [generate_random_saturated_color(i) for i in range(len(missions))]
    },
        crs=scan_areas.crs)

    process_progress_bar.progress(100, text="Finalizing")
    time.sleep(1.0)
    process_progress_bar.empty()

    return (
        project_df(project_to_wgs84, scan_areas),
        project_df(project_to_wgs84, launch_points),
        project_df(project_to_wgs84, passes),
        project_df(project_to_wgs84, passes_crosshatch),
        project_df(project_to_wgs84, missions)
    )


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

    max_mission_duration = st.number_input(
        "Target mission duration, in seconds:", min_value=0, value=20*60)
    st.markdown(
        f"Mission duration = {max_mission_duration:.0f}s = {max_mission_duration/60:.0f}min{max_mission_duration%60:.0f}s")

    corridor_width = st.number_input(
        "Width of mission corridors, in meters:", min_value=100, value=800)
    st.markdown(
        f"Width = {corridor_width:.0f}m = {corridor_width*3.28084:.0f}ft = {corridor_width*1.09361:.0f}yd = {corridor_width/1609:.2f} Miles = {corridor_width/1852:.2f} Nautical Miles")

    corridor_direction = st.number_input(
        "Direction of mission corridors, in degrees (0=North, 90=East):", min_value=0, max_value=360, value=90)
    st.markdown(
        f"Corridor Direction = {corridor_direction:.0f}deg")

    pass_spacing = st.number_input(
        "Spacing between passes in meters:", min_value=10, value=100)
    st.markdown(
        f"Spacing = {pass_spacing:.0f}m = {pass_spacing*3.28084:.0f}ft = {pass_spacing*1.09361:.0f}yd = {pass_spacing/1609:.2f} Miles = {pass_spacing/1852:.2f} Nautical Miles")

    pass_direction = st.number_input(
        "Direction of mission passes, in degrees (0=North, 90=East):", min_value=0, max_value=360, value=0)
    st.markdown(
        f"Passes Direction = {pass_direction:.0f}deg")

    pass_crosshatch = st.checkbox(
        "Crosshatch passes", value=False)
    st.markdown(
        f"Crosshatch pass = {'Yes' if pass_crosshatch else 'No'}")

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

# # Initialize session state variable
# if "show_rest" not in st.session_state:
#     st.session_state["show_rest"] = False

# # Button to trigger rendering
# if st.button("Show More"):
#     st.session_state["show_rest"] = True  # Update session state

# # Conditionally render the rest of the app
# if not st.session_state.show_rest:
#     st.stop()


(
    scan_areas,
    launch_points,
    passes,
    passes_crosshatch,
    missions
) = process(
    geojson_file,
    launch_points_df,
    scan_areas_df,
    corridor_direction,
    corridor_width,
    pass_direction,
    pass_spacing,
    pass_crosshatch,
    altitude,
    speed,
    max_mission_duration
)

with st.expander("View map of missions", expanded=True):
    m = folium.Map(location=center_coords, zoom_start=12)
    folium.GeoJson(
        scan_areas,
        style_function=lambda x: {"color": "#000000", "weight": 3},
        popup=GeoJsonPopup(
            fields=["name"],
            aliases=["Name:"],
            localize=True,
            labels=True,
            style="background-color: yellow;",
        ),
    ).add_to(m)
    folium.GeoJson(
        launch_points,
        style_function=lambda x: {"color": "#0000ff", "weight": 3},
        popup=GeoJsonPopup(
            fields=["name"],
            aliases=["Name:"],
            localize=True,
            labels=True,
            style="background-color: yellow;",
        ),
    ).add_to(m)
    # folium.GeoJson(
    #     corridors,
    #     style_function=lambda x: {"color": "#ff0000", "weight": 2}

    # ).add_to(m)

    folium.GeoJson(
        passes,
        style_function=lambda x: {"color": "#00ff00", "weight": 1}
    ).add_to(m)
    if passes_crosshatch is not None:
        folium.GeoJson(
            passes_crosshatch,
            style_function=lambda x: {"color": "#00ff00", "weight": 1}
        ).add_to(m)

    folium.GeoJson(
        missions,
        style_function=lambda x: {
            "color": x['properties']["color"], "weight": 2},
        # popup=GeoJsonPopup(
        #     fields=["name"],
        #     aliases=["Name:"],
        #     localize=True,
        #     labels=True,
        #     style="background-color: yellow;",
        # ),
    ).add_to(m)
    st_folium(m, width=700, height=500, return_on_hover=False)


###############################################################################
st.markdown("## 4. Upload Missions to Cloud")

if st.button("Upload Missions"):
    # asyncio.run(upload_missions())
    st.text("OK")
