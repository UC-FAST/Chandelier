"""Plot DOA estimates on a 3-D sphere with Plotly."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from chandelier.localizer.geometry import mic_array_positions

TEMPLATE = "presentation"
MARKER_COLOR = "#00AFFF"
MARKER_PROJ_COLOR = "#005FFF"
MICS = mic_array_positions()


class PlotlySphere:
    """Interactive Plotly figure showing DOAs on a unit sphere."""

    def __init__(self, radius: float = 1.0):
        self.radius = radius
        self._fig = go.Figure()
        self._fig.layout.template = TEMPLATE
        self._mic_pos = MICS
        self._create_sphere()

    def __call__(self, filename: str = "DOAS.html", auto_open: bool = True):
        """Write the accumulated figure to ``filename`` as standalone HTML."""

        self._fig.write_html(filename, auto_open=auto_open)

    def _create_sphere(self) -> None:
        """Draw the translucent sphere, compass arrows, and mic marks."""

        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)

        circle_r = 0.035
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = circle_r * np.cos(circle_theta)
        circle_y = circle_r * np.sin(circle_theta)
        circle_z = np.zeros_like(circle_theta)

        self._fig.add_surface(
            x=x,
            y=y,
            z=z,
            opacity=0.05,
            colorscale="Plotly3",
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True,
            ),
            hoverinfo="skip",
            name="sphere",
        )

        arrows = [
            go.Scatter3d(
                x=[0, 0.2], y=[0, 0], z=[0, 0],
                mode="lines+text",
                line=dict(color="gray", width=4),
                text="0°",
            ),
            go.Scatter3d(
                x=[0, 0], y=[0, 0.2], z=[0, 0],
                mode="lines+text",
                line=dict(color="gray", width=4),
                text="90°",
            ),
            go.Scatter3d(
                x=[0, -0.2], y=[0, 0], z=[0, 0],
                mode="lines+text",
                line=dict(color="gray", width=4),
                text="180°",
            ),
            go.Scatter3d(
                x=[0, 0], y=[0, -0.2], z=[0, 0],
                mode="lines+text",
                line=dict(color="gray", width=4),
                text="270°",
            ),
        ]

        self._fig.add_traces(
            [
                go.Scatter3d(
                    x=circle_x, y=circle_y, z=circle_z,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="mic_array1",
                ),
                go.Scatter3d(
                    x=self._mic_pos[:, 0],
                    y=self._mic_pos[:, 1],
                    z=self._mic_pos[:, 2],
                    mode="markers+text",
                    marker=dict(color=["darkseagreen"] * 4, size=5),
                    text=["mic1", "mic2", "mic3", "mic4"],
                    textposition="top center",
                    name="mic_array2",
                ),
                go.Scatter3d(
                    x=[0], y=[0], z=[1],
                    mode="markers",
                    marker=dict(color="rgb(0,95,255)", size=5),
                    visible=False,
                    name="sound_sources",
                ),
                *arrows,
            ]
        )

        self._fig.update_layout(
            title={"text": "SRP-PHAT Visualizer", "x": 0.47, "y": 0.99},
            scene_camera=dict(eye=dict(x=1.4, y=-1.4, z=1.4)),
            scene=dict(
                xaxis=dict(title="Y"),
                yaxis=dict(title="X"),
            ),
            showlegend=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
        )

    def add_doas(
        self,
        doas: np.ndarray,
        show_xyz_proj=None,
        color_marker: str = MARKER_COLOR,
        color_proj: str = MARKER_PROJ_COLOR,
    ) -> None:
        """Add one marker trace per DOA row, optionally with projections.

        ``doas`` holds ``(azimuth, elevation)`` pairs in degrees; a 1-D array
        is treated as a single row.  ``show_xyz_proj`` may be a bool to draw
        all three plane projections or a three-element mask choosing them.
        """

        if doas.ndim == 1:
            doas = np.expand_dims(doas, 0)

        # Divide by the project's truncated degrees-per-radian constant
        # (180/pi) to convert the reported angles into radians.
        z = np.sin(doas[:, 1] / 57.293)
        radius = np.cos(doas[:, 1] / 57.293)
        x = np.cos(doas[:, 0] / 57.293) * radius
        y = np.sin(doas[:, 0] / 57.293) * radius
        points = np.stack((x, y, z), axis=-1)

        scatter_list = [[points[:, 0], points[:, 1], points[:, 2], color_marker]]
        # Drop one coordinate per projection and park the result on a wall
        # of the +/-1 box so the direction stays readable in 3-D.
        xyz_plane = [
            [points[:, 0], points[:, 1], np.zeros(len(points)) - 1, color_proj],
            [points[:, 0], np.zeros(len(points)) + 1, points[:, 2], color_proj],
            [np.zeros(len(points)) - 1, points[:, 1], points[:, 2], color_proj],
        ]

        if isinstance(show_xyz_proj, list):
            for projection, plane in zip(show_xyz_proj, xyz_plane):
                if len(projection * plane) != 0:
                    scatter_list.append(projection * plane)
        elif show_xyz_proj:
            scatter_list.extend(xyz_plane)

        for px, py, pz, color in scatter_list:
            hover = (
                "X: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
                "<extra></extra>"
            )
            self._fig.add_scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="markers",
                marker=dict(color=color, size=8),
                hovertemplate=hover,
                opacity=0.05,
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the DOA visualiser."""

    parser = argparse.ArgumentParser(description="Visualise DOA estimates")
    parser.add_argument("--wave", required=True, help="CSV file with DOAs")
    parser.add_argument("--src", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render DOA traces from an exported CSV into a sibling HTML file.

    The CSV is expected to have a header row and two columns per source
    (azimuth then elevation), in source order.
    """

    args = build_parser().parse_args(argv)
    path = Path(args.wave)
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    sphere = PlotlySphere()
    # One marker/projection colour pair per source; the palettes are sized
    # for the four-source case this tool is normally used with.
    marker_colors = ["#00AFFF", "#FF0000", "#00FF00", "#FFD700"]
    projection_colors = ["#005FFF", "#BB00BB", "#00BB55", "#FFA500"]

    for index, columns in enumerate(
        [(column * 2, column * 2 + 1) for column in range(args.src)]
    ):
        # Each source occupies an (azimuth, elevation) column pair.
        doas = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=True,
            usecols=columns,
            dtype=np.float32,
        )
        sphere.add_doas(
            doas,
            show_xyz_proj=[True, False, False],
            color_marker=marker_colors[index],
            color_proj=projection_colors[index],
        )

    sphere(filename=str(path.with_suffix(".html")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
