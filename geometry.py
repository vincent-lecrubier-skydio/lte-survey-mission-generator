import streamlit as st
import json
from shapely.geometry import shape, LineString, mapping, Polygon
from shapely.ops import linemerge
import numpy as np
import pyproj
import math
from shapely.ops import transform
import uuid
from datetime import datetime
import httpx
import asyncio
import io
import zipfile

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import transform, linemerge
import pyproj


def generate_lawnmower_pattern(polygon, spacing, passes):
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

    if "North-South" in passes:
        x_coords = np.arange(minx, maxx, spacing)
        for i, x in enumerate(x_coords):
            if i % 2 == 0:
                lines.append(LineString([(x, miny), (x, maxy)]))
            else:
                lines.append(LineString([(x, maxy), (x, miny)]))

    if "East-West" in passes:
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
    if len(lawnmower_lines) > 1:
        connected_lines.append(lawnmower_lines[-1])

    merged_line = linemerge(connected_lines)

    # Convert the merged line back to WGS84 coordinates
    wgs84_merged_line = transform(project_to_wgs84, merged_line)

    if wgs84_merged_line.geom_type == 'MultiLineString' or wgs84_merged_line.geom_type == 'GeometryCollection':
        coords = [
            coord for line in wgs84_merged_line.geoms for coord in line.coords]

    else:
        coords = wgs84_merged_line.coords

    for point in coords:
        point = list(point)

    return LineString(coords)


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


def cleanup_names(df):
    # Ensure all features have a 'name' property
    if 'name' not in df.columns:
        df['name'] = None  # Create the column if it doesn't exist

    # Fill missing 'name' values with "{Geometry_type} {index}"
    df['name'] = df.apply(
        lambda row: row['name'] if row['name'] else f"{row.geometry.geom_type} {row.name}",
        axis=1
    )
    return df


def stgeodataframe(df):
    """
    Display a geodataframe as a table in a streamlit app
    """
    dftodisplay = df.copy()
    if hasattr(dftodisplay, "columns"):
        if "geometry" in dftodisplay.columns:
            dftodisplay["geometry"] = dftodisplay["geometry"].apply(
                lambda x: x.wkt)
    # st.write(dftodisplay.shape)
    # st.write(dftodisplay.columns)
    st.dataframe(dftodisplay)