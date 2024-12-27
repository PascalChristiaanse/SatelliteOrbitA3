import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


class OrbitPlotter:
    def __init__(self):
        """Initialize the plotter with 1 3D plot and 5 2D plots."""
        # Create the main figure with two columns
        self.fig = make_subplots(
            rows=1,
            cols=1,  # 6 columns: 1 for the 3D plot and 5 for the 2D plots
            specs=[
                [{"type": "surface"}]
            ],  # 3D plot in the first column, 2D in the others
            subplot_titles=["3D Earth Model"],
        )
        # self.create_earth()

    def create_earth(self, row=1, col=1):
        """Generate the 3D Earth model and add it to the subplot."""
        a = 6378  # Equatorial radius in km
        c = 6357  # Polar radius in km
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = a * np.outer(np.cos(u), np.sin(v))
        y = a * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones(np.size(u)), np.cos(v))

        self.fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="earth",
                opacity=0.7,
                showscale=False,
                name="Earth Model",
            ),
            row=row,
            col=col,
        )

    def add_orbit(self, r_list, row=1, col=1, color="#2B4162", name="Orbit path"):
        """Add a 3D orbit to the 3D Earth plot."""
        self.fig.add_trace(
            go.Scatter3d(
                x=[x[0] for x in r_list],
                y=[x[1] for x in r_list],
                z=[x[2] for x in r_list],
                mode="lines",
                line=dict(color=color, width=4),
                name=name,
            ),
            row=row,
            col=col,
        )

    def add_line_plot(self, x, y, name, color, row=1, col=2):
        """Add a simple line plot (sine wave) to the 2D subplot."""
        self.fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", line=dict(color=color, width=2), name=name
            ),
            row=row,
            col=col,
        )

    def plot(self):
        """Generate the 6 subplots (1 3D plot and 5 2D plots)."""
        # Add the 3D Earth model
        # self.create_earth(row=1, col=1)

        # Example orbit data
        # Add 2D plots

        # Update layout
        self.fig.update_layout(
            title="3D plot",
            height=900,
            margin=dict(l=10, r=10, b=10, t=50),
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    zeroline=False,
                    range=[-20000, 20000],  # Range for the X axis
                ),
                yaxis=dict(
                    showgrid=True,
                    zeroline=False,
                    range=[-20000, 20000],  # Range for the Y axis
                ),
                zaxis=dict(
                    showgrid=True,
                    zeroline=False,
                    range=[
                        -20000,
                        20000,
                    ],  # Range for the Z axis (to match the polar radius)
                ),
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
            ),
        )

        # Show the combined plot
        self.fig.show()


def main():
    plotter = OrbitPlotter()
    plotter.plot()


if __name__ == "__main__":
    main()
