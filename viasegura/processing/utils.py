def decimal_coords(coords, ref):
    decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -1 * decimal_degrees
    return decimal_degrees
