import numpy as np


def total_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points (lat1, lon1) and (lat2, lon2) in meters.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance between the two points in meters.
    """

    R = 6373.0  # approximate radius of earth in km.

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c * 1000
    return distance


def decimal_coords(coords, ref):
    """Converts GPS coordinates in degrees, minutes, seconds to decimal degrees.

    Args:
        coords (list): List of floats representing the degrees, minutes and seconds.
        ref (str): String indicating the reference for the sign of the coordinates.

    Returns:
        float: Decimal degrees equivalent of the input GPS coordinates.
    """
    decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -1 * decimal_degrees
    return decimal_degrees
