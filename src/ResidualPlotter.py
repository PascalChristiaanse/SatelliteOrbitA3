import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


class ResidualPlotter:
    def __init__(self):
        """Initialize the plotter with 1 3D plot and 5 2D plots."""
        # Create the main figure with two columns
        self.fig = make_subplots(
            rows=1,
            cols=1,  # 3 columns: 1 for the 3D plot and 5 for the 2D plots
            specs=[
                [{"type": "xy", "secondary_y": True}] * 1
            ],  # 3D plot in the first column, 2D in the others
            subplot_titles=[
                "Residuals",
                ""
            ],
        )

    def add_line_plot(self, x, y, name, color, row=1, col=1, secondary_y=False):
        if secondary_y:
            # Use secondary Y-axis if specified
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=name,
                ),
                row=row,
                col=col,
                secondary_y=True, # Specify secondary Y-axis
            )
        else:
            # Use primary Y-axis
            self.fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="lines", line=dict(color=color, width=2), name=name
                ),
                row=row,
                col=col,
            )

    def plot(self):
        # Update layout
        self.fig.update_layout(
            title="Residuals",
            height=900,
            margin=dict(l=10, r=10, b=10, t=50),
            xaxis_title="Time [s - 959300000]",
            yaxis_title="Difference [m]",
            # xaxis2_title="Time (Modified Julian date)",
            yaxis2_title="Distance [m]",
            # xaxis3_title="Time (Modified Julian date)",
            # yaxis3_title="Longitude Delta",
            xaxis=dict(tickformat=".1f",tickfont = dict(size=30)),
            font=dict(
                size=30,  # Set the font size here
            )
            # xaxis2=dict(tickformat=".1f"),
            # xaxis3=dict(tickformat=".1f"),
        )
        
        # Show the combined plot
        self.fig.show()


def main():
    plotter = ResidualPlotter()

    # Example data
    time = np.linspace(0, 10, 100)
    radial_residuals = np.sin(time)
    radial_residuals2 = np.cos(time)

    # Add line plots with one using the secondary Y axis
    plotter.add_line_plot(
        time, radial_residuals, "Radial Residuals (Primary Y)", color="blue"
    )
    plotter.add_line_plot(
        time,
        radial_residuals2,
        "Radial Residuals (Secondary Y)",
        color="red",
        secondary_y=True,
    )
    plotter.plot()


if __name__ == "__main__":
    main()
