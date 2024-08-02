import streamlit as st
import json
from shapely.geometry import shape, LineString, mapping
import numpy as np


def generate_lawnmower_pattern(polygon, spacing):
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds

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

    lawnmower_lines = [line.intersection(polygon) for line in lines]
    lawnmower_lines = [
        line for line in lawnmower_lines if not line.is_empty and line.geom_type == 'LineString']

    return lawnmower_lines


file = st.file_uploader("Upload a file", type=["json", "geojson"])
if file is not None:
    content = file.read()
    geojson = json.loads(content)
    polygon = shape(geojson.get('features')[0].get('geometry'))
    lawnmower_pattern = generate_lawnmower_pattern(polygon, spacing=0.02)
    result_geojson = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(line)} for line in lawnmower_pattern]
    }
    st.code(json.dumps(result_geojson, indent=2), language='json')
