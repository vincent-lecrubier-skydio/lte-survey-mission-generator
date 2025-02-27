# mypy: allow-untyped-defs

import enum
import io
import math
import typing as T
import os
import numpy as np
import requests
from mapbox import Geocoder
from PIL import Image

mapbox_access_token = os.environ['MAPBOX_TOKEN']
mapbox_host = "api.mapbox.com"


class GeocoderSingletonException(Exception):
    """Exception raised for errors in the GeocoderSingleton class."""

    pass


class MapboxNotConnectableException(Exception):
    """Mapbox not configured to be connectable."""

    pass


class AddressType(enum.Enum):
    """Enum for address types"""

    SIMPLE = "simple"
    FULL = "full"


class GeocoderSingleton:
    """
    Singleton class for Mapbox Geocoder

    Usage:
        geocoder = GeocoderSingleton()
        address = geocoder.reverse_geocode(longitude, latitude)

    Example:
        GeocoderSingleton().reverse_geocode(-122.332352, 37.534321)
    """

    _instance: T.Any = None

    def __new__(cls, access_token=None, host=None):
        # if not app.config["MAPBOX_CONNECTABLE"]:
        #     # On-Prem deployments by default should not hit Mapbox
        #     # Prod should not be hitting this case
        #     raise MapboxNotConnectableException()

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize the Geocoder instance
            if not access_token:
                # if no token provided explicitly, fall back to app config
                access_token = mapbox_access_token
            if not host:
                host = mapbox_host
            cls._instance.geocoder = Geocoder(
                access_token=access_token, host=host)  # type: ignore[attr-defined]
        return cls._instance

    def reverse_geocode(self, longitude, latitude) -> T.Dict[AddressType, str]:
        """
        Returns a dictionary of address types (simple and full) for a given longitude and latitude
        """
        response = self._instance.geocoder.reverse(
            lon=longitude, lat=latitude)  # type: ignore[attr-defined]
        addresses = {}

        # Check if the request was successful
        if response.status_code == 200:
            features = response.json()["features"]
            if not features:
                raise GeocoderSingletonException(
                    f"Geocoder Error: No features found for {longitude}, {latitude}"
                )

            # Currently using API V5: https://docs.mapbox.com/api/search/geocoding-v5/#geographical-feature-types
            # In API V6, use feature.address.name (combines address_number and street)
            # TODO(marvin, nacho, david, jack): Doesn't appear V6 support is coming to python client.
            # URL structure changed - need to convert to REST if we want to do upgrade
            feature = features[0]
            if (
                "address" in feature
                and "text" in feature
                and "place_type" in feature
                and "address" in feature["place_type"]
            ):
                house_number = feature.get("address", "")
                street_name = feature.get("text", "")
                addresses[AddressType.SIMPLE] = f"{house_number} {street_name}".strip(
                )

            addresses[AddressType.FULL] = features[0]["place_name"]
            return addresses

        else:
            raise GeocoderSingletonException(
                f"Geocoder Error: {response.status_code}")

    @classmethod
    def reset_instance(cls):
        cls._instance = None


EPSILON = 1e-14


def reverse_geocode(lat, lon):
    """
    Reverse geocode a latitude and longitude using Mapbox API v6.
    """
    try:
        addresses = GeocoderSingleton().reverse_geocode(lon, lat)
        if AddressType.SIMPLE in addresses:
            return addresses[AddressType.SIMPLE]
        elif AddressType.FULL in addresses:
            return addresses[AddressType.FULL]
        else:
            return "Address not found"
    except Exception as e:
        return f"Address not found: {str(e)}"
    # url = f"https://api.mapbox.com/geocoding/v6/mapbox.places/{lon},{lat}.json"
    # url = f"https://api.mapbox.com/search/geocode/v6/reverse?longitude={lon}&latitude={lat}"
    # params = {
    #     'access_token': mapbox_token,
    #     'types': 'address,place',  # Specify the types of places to include
    #     'limit': 1                # Number of results to return
    # }
    # response = requests.get(url, timeout=20, params=params)
    # if response.status_code == 200:
    #     try:
    #         data = response.json()
    #         if data.get('features') and isinstance(data['features'], list) and len(data['features']) > 0:
    #             properties = data['features'][0].get('properties', {})
    #             context = properties.get('context', {})

    #             if isinstance(context, dict):
    #                 address_name = context.get('address', {}).get('name')
    #                 place_name = context.get('place', {}).get('name')

    #                 if address_name and place_name:
    #                     return f"{address_name}, {place_name}"

    #             return properties.get('name', "Address not found")
    #         else:
    #             return "Address not found"
    #     except (ValueError, KeyError, TypeError):
    #         return "Invalid response format"
    # else:
    #     return "Address error"


class InvalidLatitudeError(Exception):
    """Raised when math errors occur beyond ~85 degrees N or S"""


class MapboxTile:
    """Abstraction over raster tiles from Mapbox"""

    def __init__(self, x: float, y: float, zoom: int):
        self.x = int(x)
        self.y = int(y)
        self.zoom = zoom

    @property
    def key(self) -> str:
        """Get tile key in Mapbox format"""
        return f"{self.zoom}/{self.x}/{self.y}"

    # Ported from https://github.com/mapbox/mercantile/blob/5975e1c0e1ec58e99f8e5770c975796e44d96b53/mercantile/__init__.py#L382
    @staticmethod
    def _xy(lng: float, lat: float) -> T.Tuple[float, float]:
        x = lng / 360.0 + 0.5
        sinlat = math.sin(math.radians(lat))

        try:
            y = 0.5 - 0.25 * \
                math.log((1.0 + sinlat) / (1.0 - sinlat)) / math.pi
        except (ValueError, ZeroDivisionError):
            raise InvalidLatitudeError(
                "Y can not be computed: lat={!r}".format(lat))
        else:
            return x, y

    # Ported from https://github.com/mapbox/mercantile/blob/5975e1c0e1ec58e99f8e5770c975796e44d96b53/mercantile/__init__.py#L398
    @staticmethod
    def from_lon_lat(lon: float, lat: float, zoom: int) -> T.Tuple["MapboxTile", float, float]:
        """
        Get the tile containing a longitude and latitude

        Parameters
        ----------
        lng, lat : float
            A longitude and latitude pair in decimal degrees.
        zoom : int
            The web mercator zoom level.
        truncate : bool, optional
            Whether or not to truncate inputs to limits of web mercator.

        Returns:
        -------
        Tile

        """

        x, y = MapboxTile._xy(lon, lat)
        Z2 = math.pow(2, zoom)

        if x <= 0:
            xtile = 0
            xoffset: float = 0
        elif x >= 1:
            xtile_float = Z2 - 1
            xtile = int(xtile_float)
            xoffset = xtile_float - xtile
        else:
            # To address loss of precision in round-tripping between tile
            # and lng/lat, points within EPSILON of the right side of a tile
            # are counted in the next tile over.
            xcalc = (x + EPSILON) * Z2
            xtile = int(math.floor(xcalc))
            xoffset = xcalc - xtile

        if y <= 0:
            ytile = 0
            yoffset: float = 0
        elif y >= 1:
            ytile_float = Z2 - 1
            ytile = int(ytile_float)
            yoffset = ytile_float - ytile
        else:
            ycalc = (y + EPSILON) * Z2
            ytile = int(math.floor(ycalc))
            yoffset = ycalc - ytile

        return (MapboxTile(xtile, ytile, zoom), xoffset, yoffset)


class ElevationProbeSingletonException(Exception):
    """Exception raised for errors in the ElevationProbeSingleton class."""

    pass


class ElevationProbeSingleton:
    """
    Singleton class for querying elevation data from Mapbox terrain-rgb tiles

    Usage:
        probe = ElevationProbeSingleton()
        elevation = probe.get_elevation(longitude, latitude)

    Example:
        ElevationProbeSingleton().get_elevation(-122.332352, 37.534321)
    """

    _instance: T.Any = None
    DEFAULT_ZOOM = 14
    access_token: str
    cache: dict

    def __new__(cls, access_token=None):
        # if not app.config["MAPBOX_CONNECTABLE"]:
        #     # On-Prem deployments by default should not hit Mapbox
        #     # Prod should not be hitting this case
        #     raise MapboxNotConnectableException()

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize the probe
            if not access_token:
                access_token = mapbox_access_token
            cls._instance.access_token = access_token
            cls._instance.cache = {}
        return cls._instance

    def _read_elevation(
        self, image_data: np.ndarray, width: int, x_offset: float, y_offset: float
    ) -> float:
        """Read elevation value from image data at tile coordinates"""
        x = int(x_offset * width)
        y = int(y_offset * width)  # Assuming square tiles

        # Get RGB values
        r, g, b = image_data[y, x]

        # Convert to elevation using Mapbox formula
        return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

    def _load_tile(self, tile: MapboxTile) -> tuple[np.ndarray, int, int]:
        """Load and parse terrain-rgb tile from Mapbox"""
        url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{tile.key}.pngraw"
        params = {"access_token": self.access_token}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Load image data
            img = Image.open(io.BytesIO(response.content))
            img = img.convert("RGB")
            width, height = img.size

            # Convert to numpy array for efficient processing
            image_data = np.array(img)

            return image_data, width, height

        except requests.exceptions.RequestException as e:
            raise ElevationProbeSingletonException(
                f"Failed to load terrain tile: {str(e)}")
        except Exception as e:
            raise ElevationProbeSingletonException(
                f"Error processing terrain tile: {str(e)}")

    def get_elevation(self, longitude: float, latitude: float, zoom: int = DEFAULT_ZOOM) -> float:
        """
        Query elevation at given coordinates

        Args:
            longitude: Longitude coordinate
            latitude: Latitude coordinate
            zoom: Zoom level (max precision at zoom 15)

        Returns:
            Elevation in meters

        Raises:
            ElevationProbeSingletonException: If elevation query fails
        """
        try:
            tile, x_offset, y_offset = MapboxTile.from_lon_lat(
                longitude, latitude, zoom)

            # Check cache first
            if tile.key not in self.cache:
                self.cache[tile.key] = self._load_tile(tile)

            image_data, width, _ = self.cache[tile.key]
            return self._read_elevation(image_data, width, x_offset, y_offset)

        except Exception as e:
            raise ElevationProbeSingletonException(
                f"Elevation query failed: {str(e)}")

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance"""
        cls._instance = None


# def mock_elevation_image_data_request(mock_get: mock.MagicMock) -> None:
#     mock_image_data = Image.new("RGBA", (256, 256), (128, 128, 128, 255))
#     mock_image_bytes = io.BytesIO()
#     mock_image_data.save(mock_image_bytes, format="PNG")
#     mock_image_bytes.seek(0)

#     mock_response = mock.Mock()
#     mock_response.status_code = 200
#     mock_response.content = mock_image_bytes.read()

#     # Set the mocked response for requests.get
#     mock_get.return_value = mock_response
