import streamlit as st
import json
from shapely.geometry import shape, LineString, mapping, Polygon
from shapely.ops import linemerge
import numpy as np
import pyproj
import math
from shapely.ops import transform


def generate_lawnmower_pattern(polygon, spacing):
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds

    # Create a local projection centered on the polygon
    local_utm = pyproj.Proj(proj='utm', zone=int(
        (minx + maxx) / 2 // 6) + 31, ellps='WGS84')
    wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
    project_to_utm = pyproj.Transformer.from_proj(wgs84, local_utm).transform
    project_to_wgs84 = pyproj.Transformer.from_proj(local_utm, wgs84).transform

    # Convert the polygon to the local UTM coordinates
    utm_polygon = transform(project_to_utm, polygon)

    # Get the bounds in the local UTM coordinates
    minx, miny, maxx, maxy = utm_polygon.bounds

    lines = []

    x_coords = np.arange(minx, maxx, spacing)
    for i, x in enumerate(x_coords):
        if i % 2 == 0:
            lines.append(LineString([(x, miny), (x, maxy)]))
        else:
            lines.append(LineString([(x, maxy), (x, miny)]))

    y_coords = np.arange(miny, maxy, spacing)
    for i, y in enumerate(y_coords):
        if i % 2 == 0:
            lines.append(LineString([(minx, y), (maxx, y)]))
        else:
            lines.append(LineString([(maxx, y), (minx, y)]))

    lawnmower_lines = [line.intersection(utm_polygon) for line in lines]
    lawnmower_lines = [
        line for line in lawnmower_lines if not line.is_empty and line.geom_type == 'LineString']

    # Connect the lines
    connected_lines = []
    for i in range(len(lawnmower_lines) - 1):
        connected_lines.append(lawnmower_lines[i])
        start_point = lawnmower_lines[i].coords[-1]
        end_point = lawnmower_lines[i + 1].coords[0]
        connected_lines.append(LineString([start_point, end_point]))
    connected_lines.append(lawnmower_lines[-1])

    merged_line = linemerge(connected_lines)

    # Convert the merged line back to WGS84 coordinates
    wgs84_merged_line = transform(project_to_wgs84, merged_line)

    return wgs84_merged_line


def split_polygon_into_squares(polygon, size):
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds

    # Create a local projection centered on the polygon
    local_utm = pyproj.Proj(proj='utm', zone=int(
        (minx + maxx) / 2 // 6) + 31, ellps='WGS84')
    wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
    project_to_utm = pyproj.Transformer.from_proj(wgs84, local_utm).transform
    project_to_wgs84 = pyproj.Transformer.from_proj(local_utm, wgs84).transform

    # Convert the polygon to the local UTM coordinates
    utm_polygon = transform(project_to_utm, polygon)

    # Get the bounds in the local UTM coordinates
    minx, miny, maxx, maxy = utm_polygon.bounds

    squares = []
    x_coords = np.arange(minx, maxx, size)
    y_coords = np.arange(miny, maxy, size)

    for x in x_coords:
        for y in y_coords:
            square = Polygon([
                (x, y),
                (x + size, y),
                (x + size, y + size),
                (x, y + size),
                (x, y)
            ])
            intersection = utm_polygon.intersection(square)
            if not intersection.is_empty:
                squares.append(transform(project_to_wgs84, intersection))

    return squares


file = st.file_uploader(
    "Upload a geojson file (from geojson.io) with one polygon of the entire city boundary that you want to scan", type=["json", "geojson"])

st.markdown("We split that polygon into many separate squares, each representing a mission to fly")
square_size = st.number_input(
    "Size of mission squares, in meters", min_value=100, value=800)
st.markdown(
    f"Square size = {square_size:.0f}m = {square_size*3.28084:.0f}ft = {square_size*1.09361:.0f}yd = {square_size/1609:.2f} Miles = {square_size/1852:.2f} Nautical Miles")
st.markdown("For each square, we generate a lawnmower pattern to scan the area with a given spacing between passes")
spacing = st.number_input(
    "Spacing between passes in meters", min_value=10, value=50)
st.markdown(
    f"Spacing = {spacing:.0f}m = {spacing*3.28084:.0f}ft = {spacing*1.09361:.0f}yd = {spacing/1609:.2f} Miles = {spacing/1852:.2f} Nautical Miles")

speed = st.number_input(
    "Flight speed in meters per second", min_value=0, value=14)
st.markdown(
    f"Flight speed = {speed:.0f}m/s = {speed*3.6:.1f}km/h = {speed*3600/1609:.1f}Mph = {speed*3600/1852:.1f}kts")


cost_fixed = st.number_input(
    "Fixed base cost for the whole program", min_value=0, value=5000)

cost_per_flight = st.number_input(
    "Fixed cost per flight mission", min_value=0, value=100)

cost_per_flight_hour = st.number_input(
    "Variable cost per flight hour", min_value=0, value=50)

mission_length = (  # horizontal passes
    square_size *  # length of each leg
    math.ceil(square_size/spacing)  # number of legs
    + square_size  # total length of transitions between legs
) + (  # vertical passes
    square_size *  # length of each leg
    math.ceil(square_size/spacing)  # number of legs
    + square_size  # total length of transitions between legs
) + (  # return to base, diagonal
    math.sqrt(square_size**2 + square_size**2)
)

mission_duration = mission_length/speed

if file is not None:
    content = file.read()
    geojson = json.loads(content)
    input_polygon = shape(geojson.get('features')[0].get('geometry'))

    squares = split_polygon_into_squares(input_polygon, size=square_size)

    st.markdown(f"Total number of missions to fly: {len(squares)}")
    st.markdown(
        f"Length of each mission = {mission_length:.0f}m = {mission_length*3.28084:.0f}ft = {mission_length*1.09361:.0f}yd = {mission_length/1609:.2f} Miles = {mission_length/1852:.2f} Nautical Miles")
    st.markdown(
        f"Duration of each mission = {mission_duration:.0f}s = {mission_duration/60:.0f}min")
    st.markdown(
        f"Total flight hours (Conservative) = {len(squares)*mission_duration/3600:.0f}h")
    st.markdown(
        f"Total cost = ${cost_fixed + (len(squares) * (cost_per_flight + (mission_duration/3600)* cost_per_flight_hour)) :,.0f}")

    with st.expander("See geojson with all mission squares"):
        result_squares_geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"mission_number": i+1}, "geometry": mapping(square)} for i, square in enumerate(squares)]
        }
        st.code(json.dumps(result_squares_geojson, indent=2), language='json')

    with st.expander("See geojson for a given mission"):
        input_square_index = st.number_input("Index of mission to inspect",
                                             min_value=1, max_value=len(squares), value=1)
        input_square = squares[input_square_index-1]
        lawnmower_pattern = generate_lawnmower_pattern(
            input_square, spacing=spacing)
        result_geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(lawnmower_pattern)}]
        }
        st.code(json.dumps(result_geojson, indent=2), language='json')
