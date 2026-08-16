import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

FILE_PATH = "puffing of the cheeks.csv"
SAVE_PATH = "puffing of the cheeks.pdf"
NUM_LANDMARKS = 68
FIRST_INTERNAL_LANDMARK = 17
DISTANCE_THRESHOLD = 300


def main():
    """Render the first CSV landmark row [68,2] as a graph."""
    data = pd.read_csv(FILE_PATH)
    x_names = [f"x_{i}" for i in range(NUM_LANDMARKS)]
    y_names = [f"y_{i}" for i in range(NUM_LANDMARKS)]
    x_values = np.asarray(data[x_names].values)
    y_values = np.asarray(data[y_names].values)
    x_values -= np.min(x_values)
    y_values -= np.min(y_values)
    internal_indices = list(range(FIRST_INTERNAL_LANDMARK, NUM_LANDMARKS))
    internal_x_values = x_values[0][internal_indices]
    internal_y_values = y_values[0][internal_indices]

    figure = plt.figure(figsize=(10, 10))
    plt.scatter(internal_x_values, internal_y_values, color="cyan", edgecolor="black", s=300)
    internal_points = np.column_stack((internal_x_values, internal_y_values))
    triangulation = Delaunay(internal_points)

    for simplex in triangulation.simplices:
        p1 = internal_points[simplex[0]]
        p2 = internal_points[simplex[1]]
        p3 = internal_points[simplex[2]]
        for edge in ((p1, p2), (p2, p3), (p3, p1)):
            if np.linalg.norm(edge[0] - edge[1]) < DISTANCE_THRESHOLD:
                plt.plot(
                    [edge[0][0], edge[1][0]],
                    [edge[0][1], edge[1][1]],
                    color="black",
                    linewidth=2,
                    linestyle="-",
                )

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(SAVE_PATH, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
