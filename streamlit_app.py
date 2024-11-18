from datetime import datetime
import json
from typing import Union
import uuid
from shapely import LineString
from shapely.geometry import mapping
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
import altair as alt
import pyproj
import colorsys
import math
import warnings
import requests

warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
mapbox_token = "pk.eyJ1Ijoic2t5ZGlvLXRlYW0iLCJhIjoiY20zbW9mYmZ0MGpsMTJpcHl3bWhsbm5rcSJ9.2bTFNXUX0RrFKLiH8EgW_g"


def reverse_geocode(lat, lon):
    """
    Reverse geocode a latitude and longitude using Mapbox API v6.
    """
    # url = f"https://api.mapbox.com/geocoding/v6/mapbox.places/{lon},{lat}.json"
    url = f"https://api.mapbox.com/search/geocode/v6/reverse?longitude={lon}&latitude={lat}"
    params = {
        'access_token': mapbox_token,
        'types': 'address,place',  # Specify the types of places to include
        'limit': 1                # Number of results to return
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['features']:
            if 'context' in data['features'][0]['properties']:
                # Extract the city and country
                return data['features'][0]['properties']['context']['address']['name'] + ", " + data['features'][0]['properties']['context']['place']['name']
            # Extract the full address
            return data['features'][0]['properties']['name']
        else:
            return "Address not found"
    else:
        return "Address error"


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


def generate_mission(linestring, index, altitude, name_template, rtx_height, rtx_speed, rtx_wait):
    with open("mission.proto.json", "r", encoding="utf8") as mission_proto_file:
        mission_proto = json.load(mission_proto_file)
        mission_proto["displayName"] = name_template.format(
            index=index, date=datetime.now().isoformat())
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
                                                "value": altitude  # m = 200ft
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
            for point in linestring.coords
        ]

        # return json string
        return json.dumps(mission_proto, indent=2)


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
    launch_points_df['address'] = launch_points_df.apply(
        lambda row: reverse_geocode(
            row.geometry.y, row.geometry.x) if row.geometry else "No geometry",
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


@st.cache_data(show_spinner=False)
def process(
        geojson_file, _launch_points_df, _scan_areas_df,
        corridor_direction, corridor_width, pass_direction, pass_spacing, crosshatch,
        altitude, speed, max_mission_duration, name_template
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

    start_slice = 0
    end_slice = slices.shape[0]

    missions_optim = []
    current_start = start_slice

    while current_start < end_slice:
        time.sleep(0.01)

        process_progress_bar.progress(30+math.floor(70*(current_start-start_slice)/(
            end_slice-start_slice)), text=f"Generating optimal slices {current_start-start_slice}/{end_slice-start_slice}")

        # Binary search to find the best range
        low, high = current_start + 1, end_slice
        best_end = None
        best_reward = float('-inf')

        while low <= high:
            mid = (low + high) // 2

            (launch_point, mission_path) = compute_total_mission_path(
                launch_points, slices, passes, passes_crosshatch, current_start, mid)
            mission_duration = compute_mission_duration(
                mission_path, altitude, speed)
            reward = mid-current_start
            # st.text(f"{current_start}, {mid}: {mission_duration}")

            if mission_duration is None:  # Invalid range, increase size
                low = mid + 1
            elif mission_duration > max_mission_duration:  # Range too large, decrease size
                high = mid - 1
            else:  # Valid range, optimize for maximum reward
                if reward > best_reward:
                    best_launch_point = launch_point.address
                    best_mission_path = mission_path
                    best_mission_duration = mission_duration
                    best_reward = reward
                    best_end = mid
                low = mid + 1  # Explore larger ranges

        # If no valid range is found, terminate the loop
        if best_end is None:
            raise ValueError(
                "No mission found for the given constraints, try making target duration larger")

        # Finalize the missions
        missions_optim.append((
            current_start,
            best_end,
            best_launch_point,
            best_mission_path,
            best_mission_duration))
        current_start = best_end + 1  # Update start for the next range

    date_now = datetime.now().isoformat(timespec='seconds')

    missions = gpd.GeoDataFrame({
        'geometry': [mission_path for (start, end, launch_point, mission_path, mission_duration) in missions_optim],
        'name': [
            name_template.format(
                index=index,
                date=date_now,
                launch_point=launch_point,
                start=start,
                end=end,
                duration=math.ceil(mission_duration)
            ) for index, (start, end, launch_point, mission_path, mission_duration)
            in enumerate(missions_optim)],
        'launch_point': [launch_point for (start, end, launch_point, mission_path, mission_duration) in missions_optim],
        'start': [start for (start, end, launch_point, mission_path, mission_duration) in missions_optim],
        'end': [end for (start, end, launch_point, mission_path, mission_duration) in missions_optim],
        'duration': [math.ceil(mission_duration) for (start, end, launch_point, mission_path, mission_duration) in missions_optim],
        'stroke': [generate_random_saturated_color(i) for i in range(len(missions_optim))],
        'stroke-width': [2 for i in range(len(missions_optim))],
        'index': [i for i in range(len(missions_optim))]
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
    "Mission name template:", value="LTE Scan | {date} | {launch_point} | #{index} | {duration}s")

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
    max_mission_duration,
    name_template
)

with st.expander("View map of missions", expanded=True):

    mission_geojson_data = gdfs_to_json(
        scan_areas, launch_points, missions).encode('utf-8')

    st.download_button(
        label="Download as GeoJSON file",
        icon="🗺️",
        data=mission_geojson_data,
        file_name="missions.geojson",
        mime="application/json",
    )

    m = folium.Map(location=center_coords, zoom_start=12)
    folium.GeoJson(
        scan_areas,
        style_function=lambda x: {"color": "#000000", "weight": 3},
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
        style_function=lambda x: {"color": "#0000ff", "weight": 3},
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
            "color": x['properties']["stroke"],
            "weight": x['properties']["stroke-width"]
        },
        popup=GeoJsonPopup(
            fields=["name", "index", "launch_point",
                    "duration", "start", "end"],
            aliases=["Mission Name", "Mission Number", "Launch Point",
                     "Duration", "Start Slice", "End Slice"],
            localize=True,
            labels=True,
            style="background-color: yellow;",
        ),
    ).add_to(m)

    st_folium(m, width=700, height=500, return_on_hover=False)

with st.expander("View missions details", expanded=False):

    mission_duration_chart = alt.Chart(missions).mark_bar().encode(
        x=alt.X("duration", bin=alt.Bin(maxbins=100, extent=[0, max_mission_duration*1.2]),
                title=f"Mission duration"),
        y=alt.Y('count()', title='Number of Missions')
    ).properties(
        title=f"Histogram of missions durations",
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

###############################################################################
st.markdown("## 4. Upload Missions to Cloud")

if st.button("Upload Missions"):
    # asyncio.run(upload_missions())
    st.text("OK")
