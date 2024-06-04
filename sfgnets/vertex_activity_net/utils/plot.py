"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Auxiliary functions for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap


def rgb_to_hex(rgb):
    """
    Convert RGB values to a hexadecimal color code.

    Args:
        rgb (tuple): A tuple containing the RGB values as floats between 0 and 1.

    Returns:
        str: Hexadecimal color code representing the RGB values.
    """
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def plotly_generate(image, max_energy=1):
    """
    Generate a 3D plot using Plotly for visualising a voxelised image.

    Args:
        image (numpy.ndarray): The voxelised image as a numpy array.
        max_energy (float, optional): The maximum energy value for color scaling. Default is 1.
    """
    # Convert the numpy array to integers
    numpy_array = image.astype(int)

    # Create a 3D figure
    fig = go.Figure()

    x1, y1, z1 = np.nonzero(numpy_array)

    # Define the cube size
    cube_size = 1

    opacity = np.clip(image / (max_energy * 0.5), 0, 1)

    base_cmap = plt.get_cmap('Wistia')
    num_colors = 100
    hex_colors = [rgb_to_hex(base_cmap(i / (num_colors - 1))) for i in range(num_colors)]
    custom_cmap = ListedColormap(hex_colors)
    N = int(max_energy) + 1  # Adjust N to your desired range
    values = np.linspace(0, N, num=N)
    colors = [rgb_to_hex(custom_cmap(value / N)) for value in values]

    # Create a mesh3d trace for each voxel (cube)
    for x, y, z in zip(x1, y1, z1):
        cube = go.Mesh3d(
            x=[x, x, x + cube_size, x + cube_size, x, x, x + cube_size, x + cube_size],
            y=[y, y + cube_size, y + cube_size, y, y, y + cube_size, y + cube_size, y],
            z=[z, z, z, z, z + cube_size, z + cube_size, z + cube_size, z + cube_size],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            colorbar=dict(
                thickness=20,
                title='Voxel Values',
            ),
            color=colors[int(image[x, y, z])],
            flatshading=True,
            opacity=opacity[x, y, z]
        )
        fig.add_trace(cube)

    # Set the aspect ratio to be equal
    fig.update_layout(scene=dict(aspectmode="cube"))

    # Set the margin to remove all margins
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Set the image size based on the aspect ratio
    width, height = 800, 800  # Adjust these values as needed
    fig.update_layout(width=width, height=height)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 5],
                       title='X',
                       showticklabels=False,
                       gridcolor="blue", backgroundcolor="#07042F"),
            yaxis=dict(range=[0, 5],
                       title='Y',
                       showticklabels=False,
                       gridcolor="blue", backgroundcolor="#07042F"),
            zaxis=dict(range=[0, 5],
                       title='Z',
                       showticklabels=False,
                       gridcolor="blue", backgroundcolor="#07042F"),
        )
    )

    # Set the camera parameters for initial zoom
    fig.update_layout(
        scene=dict(
            camera=dict(
                center=dict(x=0, y=0, z=0),  # adjust the center of the view
                eye=dict(x=1.5, y=1.5, z=1.5),  # adjust the camera's initial position
            )
        )
    )

    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')

    fig.show()
