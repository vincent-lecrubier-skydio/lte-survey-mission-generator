from shapely import MultiPolygon
import streamlit as st
import json
from shapely.geometry import shape, LineString, mapping, Polygon
from shapely.affinity import translate, rotate
import geopandas as gpd
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
from shapely.ops import transform, linemerge
import pyproj


# def generate_lawnmower_pattern(polygon, spacing, passes):
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

#     lines = []

#     if "North-South" in passes:
#         x_coords = np.arange(minx, maxx, spacing)
#         for i, x in enumerate(x_coords):
#             if i % 2 == 0:
#                 lines.append(LineString([(x, miny), (x, maxy)]))
#             else:
#                 lines.append(LineString([(x, maxy), (x, miny)]))

#     if "East-West" in passes:
#         y_coords = np.arange(miny, maxy, spacing)
#         for i, y in enumerate(y_coords):
#             if i % 2 == 0:
#                 lines.append(LineString([(minx, y), (maxx, y)]))
#             else:
#                 lines.append(LineString([(maxx, y), (minx, y)]))

#     lawnmower_lines = [line.intersection(utm_polygon) for line in lines]
#     lawnmower_lines = [
#         line for line in lawnmower_lines if not line.is_empty and line.geom_type == 'LineString']

#     # Connect the lines
#     connected_lines = []
#     for i in range(len(lawnmower_lines) - 1):
#         connected_lines.append(lawnmower_lines[i])
#         start_point = lawnmower_lines[i].coords[-1]
#         end_point = lawnmower_lines[i + 1].coords[0]
#         connected_lines.append(LineString([start_point, end_point]))
#     if len(lawnmower_lines) > 1:
#         connected_lines.append(lawnmower_lines[-1])

#     merged_line = linemerge(connected_lines)

#     # Convert the merged line back to WGS84 coordinates
#     wgs84_merged_line = transform(project_to_wgs84, merged_line)

#     if wgs84_merged_line.geom_type == 'MultiLineString' or wgs84_merged_line.geom_type == 'GeometryCollection':
#         coords = [
#             coord for line in wgs84_merged_line.geoms for coord in line.coords]

#     else:
#         coords = wgs84_merged_line.coords

#     for point in coords:
#         point = list(point)

#     return LineString(coords)


# def split_polygon_into_squares(polygon, size):
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


def cleanup_names(df):
    """
    Ensure all features have a 'name' property
    """

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


def compute_bounding_circle(polygons):
    """
    Given polygons, compute the circle that bounds all of them
    """

    minx, miny, maxx, maxy = polygons.total_bounds
    centerx, centery = ((minx+maxx)/2, (miny+maxy)/2)
    radius = np.sqrt(((minx-maxx)/2)**2+((miny-maxy)/2)**2)
    return (radius, centerx, centery)


def project_df(transformer: pyproj.Transformer, gdf):
    """
    Project a GeoDataFrame to a new coordinate reference system
    """

    if gdf is None:
        return None
    result_gdf = gdf.copy()
    result_gdf["geometry"] = result_gdf["geometry"].apply(
        lambda geom: transform(transformer.transform, geom))
    result_gdf.set_crs(transformer.target_crs,
                       inplace=True, allow_override=True)
    return result_gdf


def generate_corridors(scan_area, corridor_direction, corridor_width):
    """
    Generate corridors (Polygons) along the specified direction
    """

    radius, centerx, centery = compute_bounding_circle(scan_area)

    # Convert direction to radians
    direction_rad = np.deg2rad(-corridor_direction)

    # Define a base line along the specified direction
    base_line = LineString([
        (centerx+radius * math.cos(math.pi / 2 + direction_rad),
         centery+radius * math.sin(math.pi / 2 + direction_rad)),
        (centerx-radius * math.cos(math.pi / 2 + direction_rad),
         centery-radius * math.sin(math.pi / 2 + direction_rad))
    ])

    # Number of corridors
    half_number_of_corridors = math.ceil(radius / corridor_width)

    # Generate lines on both sides of corridors
    corridor_lines = []
    for i in range(-half_number_of_corridors, half_number_of_corridors+1):
        parallel_line = translate(
            base_line,
            xoff=i*corridor_width*math.cos(direction_rad),
            yoff=i*corridor_width*math.sin(direction_rad)
        )
        corridor_lines.append(parallel_line)

    # Convert corridor lines into scan_area (corridors)
    corridor_polygons = []
    for i in range(len(corridor_lines) - 1):
        corridor_polygons.append(
            Polygon([*corridor_lines[i].coords, *corridor_lines[i + 1].coords[::-1]]))

    # Split scan_area into corridors
    resulting_corridors = []
    for poly in scan_area.geometry:
        for corridor in corridor_polygons:
            clipped = poly.intersection(corridor)
            if not clipped.is_empty:
                resulting_corridors.append(clipped)

    # Create a GeoDataFrame of the resulting corridor scan_area
    result_gdf = gpd.GeoDataFrame(
        geometry=resulting_corridors, crs=scan_area.crs)

    return result_gdf


def generate_passes(scan_area, passes_direction, passes_spacing):
    """
    Generate passes (Lines) along the specified direction
    """

    radius, centerx, centery = compute_bounding_circle(scan_area)

    # Convert direction to radians
    direction_rad = np.deg2rad(-passes_direction)

    # Define a base line along the specified direction
    base_line = LineString([
        (centerx+radius * math.cos(math.pi / 2 + direction_rad),
         centery+radius * math.sin(math.pi / 2 + direction_rad)),
        (centerx-radius * math.cos(math.pi / 2 + direction_rad),
         centery-radius * math.sin(math.pi / 2 + direction_rad))
    ])

    # Number of passes
    half_number_of_passes = math.ceil(radius / passes_spacing)

    # Generate lines on both sides of passes
    passe_lines = []
    for i in range(-half_number_of_passes, half_number_of_passes+2):
        parallel_line = translate(
            base_line,
            xoff=(i-0.5)*passes_spacing*math.cos(direction_rad),
            yoff=(i-0.5)*passes_spacing*math.sin(direction_rad)
        )
        passe_lines.append(parallel_line)

    # Split scan_area into passes
    resulting_passes = []
    for poly in scan_area.geometry:
        for passe in passe_lines:
            clipped = poly.intersection(passe)
            if not clipped.is_empty:
                resulting_passes.append(clipped)

    # Create a GeoDataFrame of the resulting pass
    result_gdf = gpd.GeoDataFrame(
        geometry=resulting_passes, crs=scan_area.crs)

    return result_gdf


def move_mask_square(mask_square_template, direction_rad, centerx, centery, radius, iteration_spacing, index):
    """
    Given the mask square template (A square aligned with corridors, covering the whole areas),
    Move the mask square to the specified index of the iteration, covering more and more of the corridor
    """
    mask_square = rotate(
        translate(
            mask_square_template,
            xoff=2*radius - iteration_spacing*index,
            yoff=0
        ),
        direction_rad,
        origin=(centerx, centery),
        use_radians=True
    )
    return mask_square


def generate_slices(rectangle, crs, corridor_width, slice_thickness):
    """
    Generate polygons within a rectangle based on corridors and slices, indexed in a lawnmower pattern.
    Computes the final polygons directly without iterating over all intermediate indices.

    :param rectangle: A shapely Polygon representing the rectangle.
    :param corridor_width: The height of each horizontal corridor.
    :param slice_thickness: The width of each vertical slice.
    :param start_index: Starting index of the slices to generate.
    :param end_index: Ending index of the slices to generate.
    :return: List of shapely Polygons for the specified indices.
    """
    # Get the bounds of the rectangle
    minx, miny, maxx, maxy = rectangle.bounds

    # Calculate the number of horizontal corridors and vertical slices
    num_corridors = math.ceil((maxy - miny) / corridor_width)
    num_slices_per_corridor = math.ceil((maxx - minx) / slice_thickness)

    # Total number of cells
    total_cells = num_corridors * num_slices_per_corridor

    # Calculate polygons directly
    result = gpd.GeoDataFrame(
        {
            "geometry": [None] * total_cells,
            "name": [None] * total_cells
        },
        crs=crs
    )
    for index in range(0, total_cells):
        # Determine the corridor and slice index
        corridor_index = index // num_slices_per_corridor
        slice_index_within_corridor = index % num_slices_per_corridor

        # Adjust slice index for the lawnmower pattern
        if corridor_index % 2 != 0:
            slice_index_within_corridor = num_slices_per_corridor - \
                1 - slice_index_within_corridor

        # Calculate coordinates for the polygon
        slice_min_x = minx + slice_index_within_corridor * slice_thickness
        slice_max_x = min(slice_min_x + slice_thickness, maxx)
        # corridor_min_y = miny + corridor_index * corridor_width
        # corridor_max_y = min(corridor_min_y + corridor_width, maxy)
        corridor_min_y = maxy - corridor_index * corridor_width
        corridor_max_y = max(corridor_min_y - corridor_width, miny)

        # Create the polygon
        poly = Polygon([
            (slice_min_x, corridor_min_y),
            (slice_max_x, corridor_min_y),
            (slice_max_x, corridor_max_y),
            (slice_min_x, corridor_max_y),
            (slice_min_x, corridor_min_y)
        ])
        result.at[index, "geometry"] = poly
        result.at[index, "name"] = f"Slice {index}"

    return result


def compute_oriented_slices(polygons, direction, corridor_width, slice_thickness):
    """
    Compute the oriented bounding box of all polygons in a GeoDataFrame.

    :param df: GeoDataFrame containing polygons.
    :param direction: Orientation angle in degrees (counterclockwise from x-axis).
    :return: Oriented bounding box as a Polygon.
    """
    # Merge all polygons into a single geometry (union)
    combined_geom = polygons.unary_union

    if isinstance(combined_geom, Polygon):
        geometries = [combined_geom]
    elif isinstance(combined_geom, MultiPolygon):
        geometries = list(combined_geom.geoms)
    else:
        raise ValueError("Input GeoDataFrame must contain valid polygons.")

    # Get the centroid of the combined geometry
    centroid = combined_geom.centroid

    # Rotate the combined geometry by the negative direction (align to x-axis)
    rotated_geom = rotate(combined_geom, 90-direction,
                          origin=centroid, use_radians=False)

    # Create the slices in rotated space
    slices = generate_slices(rotated_geom, polygons.crs, corridor_width,
                             slice_thickness)

    # Rotate the slices back to the original orientation
    slices["geometry"] = slices["geometry"].apply(
        lambda geom: rotate(geom, direction-90,
                            origin=centroid, use_radians=False)
    )
    return slices


def compute_oriented_bounding_box(polygons, direction):
    """
    Compute the oriented bounding box of all polygons in a GeoDataFrame.

    :param df: GeoDataFrame containing polygons.
    :param direction: Orientation angle in degrees (counterclockwise from x-axis).
    :return: Oriented bounding box as a Polygon.
    """
    # Merge all polygons into a single geometry (union)
    combined_geom = polygons.unary_union

    if isinstance(combined_geom, Polygon):
        geometries = [combined_geom]
    elif isinstance(combined_geom, MultiPolygon):
        geometries = list(combined_geom.geoms)
    else:
        raise ValueError("Input GeoDataFrame must contain valid polygons.")

    # Get the centroid of the combined geometry
    centroid = combined_geom.centroid

    # Rotate the combined geometry by the negative direction (align to x-axis)
    rotated_geom = rotate(combined_geom, -direction,
                          origin=centroid, use_radians=False)

    # Get the bounding box of the rotated geometry
    minx, miny, maxx, maxy = rotated_geom.bounds

    # Create the bounding box polygon in rotated space
    rotated_bbox = Polygon(
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])

    # Rotate the bounding box back to the original orientation
    oriented_bbox = rotate(rotated_bbox, direction,
                           origin=centroid, use_radians=False)

    return oriented_bbox


def compute_mission_duration(remaining_corridors, passes, passes_crosshatch, mask_square):
    pass


def generate_next_lawnmower(scan_areas, launch_points, corridor_direction, passes, passes_crosshatch, iteration_spacing):
    """
    Generate the next lawnmower pattern
    """

    # radius, centerx, centery = compute_bounding_circle(scan_areas)
    # direction_rad = np.deg2rad(180+90-corridor_direction)
    # mask_square_template = Polygon([
    #     (centerx+radius, centery+radius),
    #     (centerx+radius, centery-radius),
    #     (centerx-radius, centery-radius),
    #     (centerx-radius, centery+radius),
    # ])

    # result = move_mask_square(
    #     mask_square_template, direction_rad, centerx, centery, radius, iteration_spacing, 35)

    # result = compute_oriented_bounding_box(scan_areas, corridor_direction)
    # result_gdf = gpd.GeoDataFrame(geometry=[result], crs=scan_areas.crs)

    result_gdf = compute_oriented_slices(
        scan_areas, corridor_direction, 800, iteration_spacing)

    return result_gdf


def generate_next_mission(scan_areas, remaining_corridors, corridor_direction, passes, passes_crosshatch):
    return generate_next_lawnmower(scan_areas, remaining_corridors, corridor_direction, passes, passes_crosshatch)
