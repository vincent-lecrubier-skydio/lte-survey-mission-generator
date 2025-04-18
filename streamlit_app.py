import asyncio
from datetime import datetime
import io
import json
from typing import Union
import uuid
import httpx
from shapely import LineString, Point
from shapely.geometry import mapping

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import time
from folium.features import GeoJsonPopup
from streamlit_folium import st_folium
import altair as alt
import pyproj
import colorsys
import math
import zipfile
import warnings

from geometry import generate_oriented_slices, compute_total_mission_path, generate_passes, project_df, stgeodataframe
from mapbox_util import reverse_geocode, ElevationProbeSingleton

warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)


def generate_random_saturated_color(index):
    # Generate a random hue (0-360 degrees)
    hue = (index * 137) % 360  # A multiplier like 137 ensures a spread of hues
    # Saturation and lightness are set for high saturation and medium brightness
    saturation = 100  # Fully saturated
    lightness = 50    # Medium lightness for vivid color
    r, g, b = colorsys.hls_to_rgb(
        hue / 360.0, lightness / 100.0, saturation / 100.0)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def gdfs_to_json(*gdfs):
    features = []
    for gdf in gdfs:
        for _, row in gdf.iterrows():
            properties = {key: value for key, value in row.items(
            ) if key != 'geometry' and pd.notnull(value)}
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(row.geometry)
            }
            features.append(feature)
    result = {
        "type": "FeatureCollection",
        "features": features
    }
    geojson_str = json.dumps(result)
    return geojson_str


def simplestyle_style_function(feature):
    """
    Style function for GeoJSON Simplestyle properties.
    Parameters:
        - feature: A single GeoJSON feature.
    Returns:
        - A dictionary with style attributes for Folium.
    """
    # Extract properties
    properties = feature.get('properties', {})

    # Simplestyle properties
    stroke = properties.get('stroke', '#3388ff')  # Default blue
    stroke_width = properties.get('stroke-width', 2)
    stroke_opacity = properties.get('stroke-opacity', 1.0)
    fill = properties.get('fill', '#3388ff')  # Default blue
    fill_opacity = properties.get('fill-opacity', 0.2)

    return {
        'color': stroke,
        'weight': stroke_width,
        'opacity': stroke_opacity,
        'fillColor': fill,
        'fillOpacity': fill_opacity,
    }


def format_seconds_to_hm(seconds: float) -> str:
    """
    Format a float representing seconds into a string with hours, minutes, and seconds.

    Parameters:
        seconds (float): The number of seconds.

    Returns:
        str: The formatted string in the format 'XhYmZs'.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}h {minutes:d}m"


def compute_mission_duration(
    mission_path: LineString,
    altitude: float,
    horizontal_speed_meters_per_second: float,
    time_lost_per_waypoint_seconds=7.0,
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


def generate_mission(row, altitude, rtx_height, rtx_speed, rtx_wait):
    linestring = row.geometry
    start_altitude = linestring.coords[0][2]
    with open("mission.proto.json", "r", encoding="utf8") as mission_proto_file:
        mission_proto = json.load(mission_proto_file)
        mission_proto["displayName"] = row.get("name")
        mission_proto["templateUuid"] = str(uuid.uuid4())
        mission_proto["rtxSettings"]["minimumHeight"] = rtx_height
        mission_proto["rtxSettings"]["speed"] = rtx_speed
        mission_proto["rtxSettings"]["waitTime"] = rtx_wait
        mission_proto["actions"][0]["actionUuid"] = str(uuid.uuid4())
        mission_proto['actions'][0]['args']['sequence']["actions"] = [
            {
                "actionUuid": str(uuid.uuid4()),
                "actionKey": "Sequence",
                "args": {
                    "sequence": {
                        "name": "",
                        "actions": [
                            {
                                "actionUuid": str(uuid.uuid4()),
                                "actionKey": "SetObstacleAvoidance",
                                "args": {
                                    "setObstacleAvoidance": {
                                        "oaSetting": 1
                                    },
                                    "photoOnCompletion": False,
                                    "isSkippable": False
                                }
                            },
                            {
                                "actionUuid": str(uuid.uuid4()),
                                "actionKey": "StopVideo",
                                "args": {
                                    "stopVideo": {
                                        "noArgs": False
                                    },
                                    "photoOnCompletion": False,
                                    "isSkippable": False
                                }
                            },
                            {
                                "actionUuid": str(uuid.uuid4()),
                                "actionKey": "GotoWaypoint",
                                "args": {
                                    "gotoWaypoint": {
                                        "waypoint": {
                                            "xy": {
                                                "frame": 3,
                                                "x": point[1],
                                                "y": point[0]
                                            },
                                            "z": {
                                                "frame": 5,
                                                # SUCKS but limitation of mission api
                                                "value": max(0, min(point[2]-start_altitude, 200))
                                            },
                                            "heading": {
                                                "value": 1.5707963267948966,
                                                "frame": 3
                                            },
                                            "gimbalPitch": {
                                                "value": 0.523  # 30 deg down
                                            }
                                        },
                                        "motionArgs": {
                                            "traversalArgs": {
                                                "heightMode": 1,
                                                "speed": 16.0  # m/s = 36mph
                                            },
                                            "lookAtArgs": {
                                                "ignoreTargetHeading": False,
                                                "headingMode": 7,
                                                "ignoreTargetGimbalPitch": False,
                                                "gimbalPitchMode": 1
                                            }
                                        }
                                    },
                                    "photoOnCompletion": False,
                                    "isSkippable": False
                                }
                            },
                            {
                                "actionUuid": str(uuid.uuid4()),
                                "actionKey": "SetObstacleAvoidance",
                                "args": {
                                    "setObstacleAvoidance": {
                                        "oaSetting": 1
                                    },
                                    "photoOnCompletion": False,
                                    "isSkippable": False
                                }
                            }
                        ],
                        "hideReverseUi": False
                    },
                    "photoOnCompletion": False,
                    "isSkippable": False
                }
            }
            # For all points in the linestring except first and last (which are exactly launch point)
            for point in linestring.coords[1:-1]
        ]

        # return json string
        return json.dumps(mission_proto, indent=2)


def compute_waypoints(missions_df):
    """
    Computes a DataFrame with waypoints and their range to the first point in their respective LineStrings.

    Parameters:
    - df: GeoDataFrame containing LineStrings.

    Returns:
    - A DataFrame containing the LineString index, waypoint index, waypoint coordinates, and range
    """
    waypoint_data = []
    for idx, row in missions_df.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString' and len(geom.coords) > 2:
            first_point = geom.coords[0]
            for i, point in enumerate(geom.coords[1:-1], start=1):
                waypoint_data.append({
                    'mission_index': idx,
                    'waypoint_index': i,
                    'geometry': Point(point),
                    'range': LineString([first_point, point]).length
                })

    return gpd.GeoDataFrame(waypoint_data, geometry='geometry', crs=missions_df.crs)


@st.cache_data(show_spinner=False)
def preprocess(geojson_file) -> pd.DataFrame:
    preprocess_progress_bar = st.progress(0, text="Loading File")

    df = gpd.read_file(geojson_file)

    preprocess_progress_bar.progress(5, text="Cleaning geometries")

    # Drop rows with invalid or null geometry
    initial_count = len(df)
    df = df[df.geometry.notnull()]
    df = df[df.geometry.is_valid]
    cleaned_count = len(df)

    removed_count = initial_count - cleaned_count
    if removed_count > 0:
        st.warning(f"{removed_count} invalid or null geometries removed from the input GeoJSON.")

    # Keep only Points and Polygons
    valid_types = ["Point", "Polygon"]
    df = df[df.geometry.type.isin(valid_types)]
    if len(df) == 0:
        preprocess_progress_bar.empty()
        raise ValueError("No valid geometries (Points or Polygons) found after cleaning the input file.")

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
    if launch_points_df is None or len(launch_points_df) <= 0:
        preprocess_progress_bar.empty()
        raise ValueError("""
                No valid launch points found in the uploaded file.
                
                Suggested fixes:
                  - Add points to the geojson file, representing launch locations
            """)

    elevation_probe = ElevationProbeSingleton()
    launch_points_df['geometry'] = launch_points_df.apply(
        lambda row: Point(row.geometry.x, row.geometry.y, elevation_probe.get_elevation(
            row.geometry.x, row.geometry.y)),
        axis=1
    )

    # Fill missing 'name' values
    launch_points_df['address'] = launch_points_df.apply(
        lambda row: reverse_geocode(
            row.geometry.y, row.geometry.x) if row.geometry else "No geometry",
        axis=1
    )
    launch_points_df['name'] = launch_points_df.apply(
        lambda row: row['name'] if row['name'] else f"Launch Point {row.name}: {row['address']}",
        axis=1
    )

    preprocess_progress_bar.progress(60, text="Generating Scan Areas Names")

    # Scan Areas: Filter for Polygons (including MultiPolygons if needed)
    scan_areas_df = df[df.geometry.type == "Polygon"].copy()
    if scan_areas_df is None or len(scan_areas_df) <= 0:
        preprocess_progress_bar.empty()
        raise ValueError("""
                No valid scan areas found in the uploaded file.
                
                Suggested fixes:
                  - Add polygons to the geojson file, representing areas to scan
            """)
    # Fill missing 'name' values
    scan_areas_df['name'] = scan_areas_df.apply(
        lambda row: row['name'] if row['name'] else f"Scan Area {row.name}",
        axis=1
    )

    preprocess_progress_bar.progress(100, text="Finalizing")
    preprocess_progress_bar.empty()

    return (center_coords, launch_points_df, scan_areas_df)


@st.cache_data(show_spinner=False)
def process(
        geojson_file, _launch_points_df, _scan_areas_df,
        corridor_direction, corridor_width, pass_direction, pass_spacing, crosshatch,
        altitude, terrain_follow, speed, max_mission_duration, name_template, total_bounds, max_segment_length: int = 100,
    horizontal_tolerance: float = 1,
    vertical_tolerance: float = 10
) -> pd.DataFrame:
    process_progress_bar = st.progress(0, text="Computing projections")
    process_debug_text = st.empty()

    # Create a local projection centered on the polygon
    minx, miny, maxx, maxy = total_bounds
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
        scan_areas, pass_direction, pass_spacing, altitude, project_to_wgs84, project_to_utm)
    if crosshatch:
        # Add crosshatch perpendicular passes
        passes_crosshatch = generate_passes(
            scan_areas, pass_direction + 90, pass_spacing, altitude, project_to_wgs84, project_to_utm)
    else:
        passes_crosshatch = None

    start_slice = 0
    end_slice = slices.shape[0]

    missions_optim = []
    current_start = start_slice

    while current_start < end_slice:
        time.sleep(0.01)

        process_progress_bar.progress(30+math.floor(60*(current_start-start_slice)/(
            end_slice-start_slice)), text=f"Generating optimal slices {current_start-start_slice}/{end_slice-start_slice}")

        # Binary search to find the best range
        low, high = current_start + 1, end_slice
        best_end = None
        best_reward = float('-inf')

        # process_debug_text.text("ok1")

        while low <= high:
            mid = (low + high) // 2

            # process_debug_text.text("ok1 ok1")

            (
                launch_point,
                mission_path,
                scanned_polygon
            ) = compute_total_mission_path(
                launch_points,
                slices,
                passes,  # ok
                passes_crosshatch,
                current_start,
                mid,
                altitude,
                project_to_wgs84,
                project_to_utm,
                max_segment_length,
                horizontal_tolerance,
                vertical_tolerance
            )

            # process_debug_text.text("ok1 ok2")

            mission_duration = compute_mission_duration(
                mission_path, altitude, speed)
            reward = mid-current_start

            # process_debug_text.text("ok1 ok3")
            # st.text(f"{current_start}, {mid}: {mission_duration}")

            if mission_duration is None:  # Invalid range, increase size
                low = mid + 1
            elif mission_duration > max_mission_duration:  # Range too large, decrease size
                high = mid - 1
            else:  # Valid range, optimize for maximum reward
                if reward > best_reward:
                    best_launch_point = launch_point
                    best_mission_path = mission_path
                    best_mission_scanned_polygon = scanned_polygon
                    best_mission_duration = mission_duration
                    best_reward = reward
                    best_end = mid
                low = mid + 1  # Explore larger ranges

            # process_debug_text.text("ok1 ok4")

        # process_debug_text.text("ok2")
        # If no valid range is found, terminate the loop and report problem
        if best_end is None:
            process_progress_bar.empty()
            error = ValueError(f"""
                No mission found for the given constraints at slice {low}/{end_slice}. The problematic area (Red in map below) cannot be accessed in the required time.
                
                Suggested fixes:
                  - Make the Mission target duration longer (in Mission planning parameters)
                  - Add launch points closer to the problematic areas
                  - Make the scan area smaller
            """)
            problematic_slices = slices.iloc[low:low+1]
            problematic_slices["stroke"] = "#ff0000"
            problematic_slices["stroke-thickness"] = 3
            problematic_slices["stroke-opacity"] = 1.0
            problematic_slices["fill"] = "#ff0000"
            problematic_slices["fill-opacity"] = 0.2
            error.location = problematic_slices
            error.launch_points = project_df(project_to_wgs84, launch_points)
            error.scan_areas = project_df(project_to_wgs84, scan_areas)

            raise error

        # Finalize the missions
        missions_optim.append((
            current_start,
            best_end,
            best_launch_point,
            best_mission_path,
            scan_areas.intersection(
                best_mission_scanned_polygon).union_all(),
            best_mission_duration))
        current_start = best_end + 1  # Update start for the next range

    date_now = datetime.now().isoformat(timespec='seconds')

    missions = gpd.GeoDataFrame({
        # 'geometry':
        # [adjust_linestring_altitude_terrain_follow(mission_path, altitude,project_to_wgs84,project_to_utm) for (
        #     start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim]
        # if terrain_follow
        # else [adjust_linestring_altitude_flat(mission_path, altitude) for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'geometry': [mission_path for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'name': [
            name_template.format(
                index=index+1,
                date=date_now,
                launch_point=launch_point.name,
                start=start,
                end=end,
                duration=math.ceil(mission_duration)
            ) for index, (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration)
            in enumerate(missions_optim)],
        'launch_point': [launch_point.name for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'start': [start for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'end': [end for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'duration': [math.ceil(mission_duration) for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'range': [max(launch_point.geometry.distance(Point(coord)) for coord in mission_path.coords) for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'distance': [math.ceil(mission_duration)*speed for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'area': [round(mission_scanned_polygon.area) for (start, end, launch_point, mission_path, mission_scanned_polygon, mission_duration) in missions_optim],
        'stroke': [generate_random_saturated_color(i) for i in range(len(missions_optim))],
        'stroke-width': [2 for i in range(len(missions_optim))],
        'index': [i for i in range(len(missions_optim))]
    },
        crs=scan_areas.crs)

    process_progress_bar.progress(95, text="Computing waypoint statistics")

    waypoints = compute_waypoints(missions)

    process_progress_bar.progress(100, text="Finalizing")
    time.sleep(1.0)
    process_progress_bar.empty()

    return (
        project_df(project_to_wgs84, scan_areas),
        project_df(project_to_wgs84, launch_points),
        project_df(project_to_wgs84, passes),
        project_df(project_to_wgs84, passes_crosshatch),
        project_df(project_to_wgs84, missions),
        project_df(project_to_wgs84, waypoints)
    )


def main():
    st.set_page_config(
        page_title="LTE Survey Mission Generator",
        page_icon="📶",
        layout="wide"
    )

    ###############################################################################
    st.markdown("# 📶 LTE Survey Mission Generator")

    ###############################################################################
    st.markdown("## 1. Upload input file")

    geojson_file = st.file_uploader("""
Upload a geojson file containing: Polygons covering the area you want to scan and Points representing launch/land locations.

Use [geojson.io](https://geojson.io) to create your geojson files. Example valid geojson file: [san-mateo.geojson](/app/static/san-mateo.geojson)
    """,
                                    type=["json", "geojson"])

    if geojson_file is None:
        st.stop()

    try:
        (center_coords, launch_points_df, scan_areas_df) = preprocess(geojson_file)
    except ValueError as e:
        st.error(e)
        st.stop()

    with st.expander("View map of scan areas and launch points"):
        m = folium.Map(location=center_coords, zoom_start=12)
        folium.GeoJson(
            pd.concat([launch_points_df, scan_areas_df]),
            style_function=simplestyle_style_function,
            popup=GeoJsonPopup(
                fields=["name"],
                aliases=["Name"],
                localize=True,
                labels=True,
                style="background-color: yellow;",
            ),
        ).add_to(m)
        st_folium(m, width=700, height=500, return_on_hover=False)

    ###############################################################################
    st.markdown("## 2. Customize parameters")

    name_template = st.text_input(
        "Mission name template:", value="LTE Scan - {date} - Flight {index} - {duration}s - {launch_point}")

    with st.expander("Mission Planning Parameters"):

        st.markdown(
            """
            We generate a lawnmower pattern to scan the area with a given spacing between passes.
            We slice the scan area into separate corridors of specified width, along the first pass axis.
            For each corridor, we then slice it again into separate slices across the first pass axis.
            We compute the size of each slice such that each mission is as long as possible, but no more than target duration.
            We then fly the drone at specific speed and altitude along these passes.
            """)

        terrain_follow = st.toggle("Terrain Follow", value=True)
        # st.markdown(
        #     f"Terrain follow = {'Yes' if terrain_follow else 'No, fixed altitude above launch point'}")

        pass_crosshatch = st.toggle(
            "Crosshatch passes", value=False)
        # st.markdown(
        #     f"Crosshatch pass = {'Yes' if pass_crosshatch else 'No'}")

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
            "Fixed base cost for the whole program:", min_value=0, value=400)

        cost_per_flight = st.number_input(
            "Fixed cost per flight mission:", min_value=0, value=200)

        cost_per_flight_hour = st.number_input(
            "Variable cost per flight hour:", min_value=0, value=70)

    ###############################################################################
    st.markdown("## 3. View missions")

    try:
        total_bounds = scan_areas_df.total_bounds
        (
            scan_areas,
            launch_points,
            passes,
            passes_crosshatch,
            missions,
            waypoints
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
            terrain_follow,
            speed,
            max_mission_duration,
            name_template,
            total_bounds
        )
    except ValueError as e:
        st.error(e)

        if hasattr(e, 'location'):
            with st.expander("Map showing problematic area in red", expanded=True):
                error_geojson_data = gdfs_to_json(
                    e.location, e.launch_points, e.scan_areas).encode('utf-8')
                st.download_button(
                    label="Download GeoJSON file of problematic areas for further analysis",
                    icon="🗺️",
                    data=error_geojson_data,
                    file_name="missions.geojson",
                    mime="application/json",
                )

                m = folium.Map(location=center_coords, zoom_start=12)
                folium.GeoJson(
                    e.scan_areas,
                    style_function=simplestyle_style_function,
                    popup=GeoJsonPopup(
                        fields=["name"],
                        aliases=["Name"],
                        localize=True,
                        labels=True,
                        style="background-color: yellow;",
                    ),
                ).add_to(m)
                folium.GeoJson(
                    e.launch_points,
                    style_function=simplestyle_style_function,
                    popup=GeoJsonPopup(
                        fields=["name", "address"],
                        aliases=["Name", "Address"],
                        localize=True,
                        labels=True,
                        style="background-color: yellow;",
                    ),
                ).add_to(m)
                folium.GeoJson(
                    e.location,
                    style_function=simplestyle_style_function,
                    # popup=GeoJsonPopup(
                    #     fields=["name"],
                    #     aliases=["Name"],
                    #     localize=True,
                    #     labels=True,
                    #     style="background-color: yellow;",
                    # ),
                ).add_to(m)

                st_folium(m, width=700, height=500, return_on_hover=False)
        st.stop()

    with st.expander("Overview Metrics", expanded=True):
        col11, col12, col13 = st.columns(3)
        col11.metric("Number of flights", f"{len(missions):,d} flights",
                     help="Total number of flight missions to perform in order to scan the entire area")
        col12.metric("Total flight time", format_seconds_to_hm(
            missions["duration"].sum()), help="Sum of all mission durations, total flight time to scan the entire area")
        col13.metric("Total flight distance",
                     f"{missions['distance'].sum() / 1609.0:,.0f} miles", help="Sum of all mission distances, total flight distance to scan the entire area")

        col21, col22, col23 = st.columns(3)
        col21.metric("Total cost",
                     f"${cost_fixed + len(missions) * cost_per_flight + missions['duration'].sum() / 3600 * cost_per_flight_hour:,.0f}", help="Total cost estimate for the entire program")
        col22.metric("Total scanned area",
                     f"{missions['area'].sum()/(1609*1609):,.1f} sq mi", help="Total surface area scanned by all missions")
        col23.metric("Total waypoints",
                     f"{len(waypoints):,d}", help="Sum of number of waypoints in all missions")

        col31, col32, col33 = st.columns(3)
        col31.metric("Launch points",
                     f"{missions['launch_point'].nunique():,d} locations", help="Number of launch point locations effectively used in the missions")
        col32.metric("75th percentile range",
                     f"{waypoints['range'].quantile(0.75) / 1609.0:,.2f} miles", help="75% of flight time will be done within this distance from the launch point")
        col33.metric("Max range",
                     f"{waypoints['range'].max() / 1609.0:,.2f} miles", help="Maximum range from launch point reached during the furthest mission waypoint")

    with st.expander("Map", expanded=True):

        mission_geojson_data = gdfs_to_json(
            scan_areas, launch_points, missions).encode('utf-8')

        st.download_button(
            label="Download as annotated GeoJSON file",
            icon="🗺️",
            data=mission_geojson_data,
            file_name="missions.geojson",
            mime="application/json",
        )

        m = folium.Map(location=center_coords, zoom_start=12)
        folium.GeoJson(
            scan_areas,
            style_function=simplestyle_style_function,
            # style_function=lambda x: {"color": "#000000", "weight": 3},
            # popup=GeoJsonPopup(
            #     fields=["name"],
            #     aliases=["Name"],
            #     localize=True,
            #     labels=True,
            #     style="background-color: yellow;",
            # ),
        ).add_to(m)
        folium.GeoJson(
            launch_points,
            style_function=simplestyle_style_function,
            # style_function=lambda x: {"color": "#0000ff", "weight": 3},
            popup=GeoJsonPopup(
                fields=["name", "address"],
                aliases=["Name", "Address"],
                localize=True,
                labels=True,
                style="background-color: yellow;",
            ),
        ).add_to(m)
        # folium.GeoJson(
        #     corridors,
        #     style_function=lambda x: {"color": "#ff0000", "weight": 2}

        # ).add_to(m)

        # folium.GeoJson(
        #     passes,
        #     style_function=lambda x: {"color": "#00ff00", "weight": 1}
        # ).add_to(m)
        # if passes_crosshatch is not None:
        #     folium.GeoJson(
        #         passes_crosshatch,
        #         style_function=lambda x: {"color": "#00ff00", "weight": 1}
        #     ).add_to(m)

        folium.GeoJson(
            missions,
            # style_function=lambda x: {
            #     "color": x['properties']["stroke"],
            #     "weight": x['properties']["stroke-width"]
            # },
            style_function=simplestyle_style_function,
            popup=GeoJsonPopup(
                fields=["name", "index", "launch_point",
                        "duration", "distance", "range", "area", "start", "end"],
                aliases=["Mission Name", "Mission Number", "Launch Point",
                         "Duration(s)", "Distance(m)", "Range(m)", "Area(sq m)", "Start Slice", "End Slice"],
                localize=True,
                labels=True,
                style="background-color: yellow;",
            ),
        ).add_to(m)

        st_folium(m, width=700, height=500, return_on_hover=False)

    with st.expander("Analytics", expanded=False):

        mission_duration_chart = alt.Chart(missions).mark_bar().encode(
            x=alt.X("duration", bin=alt.Bin(maxbins=100, extent=[0, max_mission_duration*1.2]),
                    title="Mission duration (s)"),
            y=alt.Y('count()', title='Number of Missions')
        ).properties(
            title="Histogram of missions durations",
            width=600,
            height=400
        )
        st.altair_chart(mission_duration_chart, use_container_width=True)

        launch_point_chart = alt.Chart(missions).mark_bar().encode(
            x=alt.X('launch_point:N', title='Mission Launch Point', axis=alt.Axis(
                labelAngle=-45,  # Rotate labels for better readability
                labelOverlap=False,  # Avoid overlapping labels
                labelLimit=200,  # Increase the maximum length of labels
                labelAlign='right'  # Align labels to avoid truncation
            )),
            y=alt.Y('count()', title='Number of Missions'),
            tooltip=['launch_point', 'count()']  # Tooltip for more information
        ).properties(
            title="Histogram of missions per launch point",
            width=600,
            height=400
        )
        st.altair_chart(launch_point_chart, use_container_width=True)

        range_chart = alt.Chart(waypoints).mark_bar().encode(
            x=alt.X("range", bin=alt.Bin(maxbins=100, extent=[0, waypoints['range'].max()]),
                    title="Range of waypoint (m)"),
            y=alt.Y('count()', title='Number of waypoints')
        ).properties(
            title="Histogram of waypoint ranges from launch point",
            width=600,
            height=400
        )
        st.altair_chart(range_chart, use_container_width=True)

        waypoints_chart = alt.Chart(waypoints).mark_bar().encode(
            x=alt.X("mission_index", bin=alt.Bin(step=1, extent=[0, waypoints['mission_index'].max()+1]),
                    title="Mission index"),
            y=alt.Y('count()', title='Number of waypoints')
        ).properties(
            title="Number of waypoints per mission",
            width=600,
            height=400
        )
        st.altair_chart(waypoints_chart, use_container_width=True)

    with st.expander("Details", expanded=False):
        stgeodataframe(missions)

    ###############################################################################
    st.markdown("## 4. Get Missions")

    missions_protos_json = [
        generate_mission(row, altitude, rtx_height, rtx_speed, rtx_wait)
        for i, row in missions.iterrows()
    ]

    # Create an in-memory bytes buffer to hold the ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, mission_proto_json in enumerate(missions_protos_json):
            filename = f"mission_{i}.json"
            zip_file.writestr(filename, mission_proto_json)
    # Seek to the beginning of the BytesIO buffer
    zip_buffer.seek(0)
    # Provide the ZIP file for download
    st.download_button(
        label="Download zip file of all missions.json",
        icon="🗄️",
        data=zip_buffer,
        file_name="missions.zip",
        mime="application/zip"
    )

    with st.expander("Skydio Cloud settings", expanded=False):

        cloud_api_url = st.text_input(
            "Cloud API URL", value="https://api.skydio.com/api/v0/mission_document/template")
        api_key = st.text_input("API Token", type="password")

    if api_key is not None and cloud_api_url is not None and cloud_api_url != "" and api_key != "" and len(missions_protos_json) > 0:
        if st.button("Upload all missions to Skydio Cloud", icon="🚀"):
            upload_progress_bar = st.progress(0.0, text="Uploading missions")

            async def upload_mission(session, request_url, headers, mission_json_data, i):
                upload_progress_bar.progress(
                    i/len(missions_protos_json), text=f"Uploading missions ({i}/{len(missions_protos_json)})")
                response = await session.post(request_url, headers=headers, data=mission_json_data)
                if response.status_code != 200:
                    return f"Mission {i}: {response.text}"
                return None

            async def upload_missions():
                headers = {
                    "Authorization": f"{api_key}",
                    "Content-Type": "application/json"
                }
                error_messages = []
                batch_size = 50  # Number of requests allowed per second
                delay = 1  # Delay in seconds between batches

                async with httpx.AsyncClient() as session:
                    for start in range(0, len(missions_protos_json), batch_size):
                        batch = missions_protos_json[start:start + batch_size]
                        tasks = [
                            upload_mission(session, cloud_api_url,
                                           headers, mission_proto_json, i)
                            for i, mission_proto_json in enumerate(batch, start=start)
                        ]

                        results = await asyncio.gather(*tasks)

                        for result in results:
                            if result:
                                error_messages.append(result)

                        # Delay between batches
                        if start + batch_size < len(missions_protos_json):
                            await asyncio.sleep(delay)

                upload_progress_bar.progress(100, text="Finalizing")
                time.sleep(1.0)
                upload_progress_bar.empty()

                if error_messages:
                    st.error("Errors occurred during the upload:\n" +
                             "\n".join(error_messages))
                else:
                    st.success("All missions uploaded successfully!")
            asyncio.run(upload_missions())
    else:
        st.warning(
            "Please provide the Cloud API URL and API Token in Skydio Cloud settings above in order to upload missions")


main()
