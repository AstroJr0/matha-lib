# Everything except plot_polygon lives in the C extension matha/_geometry.c.
# Matplotlib is imported lazily inside plot_polygon, so `import matha` no
# longer pays for pyplot.
from ._geometry import (
    distance,
    midpoint,
    slope,
    perimeter,
    area,
    centroid,
    circle_area,
    circle_circumference,
)


def plot_polygon(points: list):
    """
    Plots the polygon defined by a list of (x, y) points using Matplotlib.

    Args:
        points: A list of (x, y) tuples.
    """
    import matplotlib.pyplot as plt

    # unzip the points into X and Y lists
    x, y = zip(*points)

    # Repeat the first point at the end to close the shape visually
    x += (x[0],)
    y += (y[0],)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', label='Vertices')
    plt.fill(x, y, alpha=0.3, label='Area')

    # plot centroid if the polygon has one
    try:
        c = centroid(list(points))
        plt.plot(c[0], c[1], 'rx', label='Centroid')
    except ValueError:
        pass

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polygon Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # ensure aspect ratio is square so circles look like circles
    plt.show()
    
