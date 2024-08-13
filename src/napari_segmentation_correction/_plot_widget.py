import os
from pathlib import Path

import matplotlib.pyplot as plt
import napari.layers
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.colors import ListedColormap, to_rgb
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

ICON_ROOT = Path(__file__).parent / "icons"


class PlotWidget(QWidget):
    """Customized plotting widget class.

    Intended for interactive plotting of features in a pandas dataframe (props) belonging to a labels layer (labels).
    """

    def __init__(self, props: pd.DataFrame, labels: napari.layers.Labels):
        super().__init__()

        self.labels = labels
        self.props = props

        # Main plot.
        self.fig = plt.figure(constrained_layout=True)
        self.plot_canvas = FigureCanvas(self.fig)
        self.ax = self.plot_canvas.figure.subplots()
        self.toolbar = NavigationToolbar2QT(self.plot_canvas)

        # Specify plot customizations.
        self.fig.patch.set_facecolor("#262930")
        self.ax.tick_params(colors="white")
        self.ax.set_facecolor("#262930")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        for action_name in self.toolbar._actions:
            action = self.toolbar._actions[action_name]
            icon_path = os.path.join(ICON_ROOT, action_name + ".png")
            action.setIcon(QIcon(icon_path))

        # Create a dropdown window for selecting what to plot on the axes.
        x_axis_layout = QHBoxLayout()
        self.x_combo = QComboBox()
        self.x_combo.addItems(
            [item for item in self.props.columns if item != "index"]
        )
        self.x_combo.setCurrentText("time_point")
        x_axis_layout.addWidget(QLabel("x-axis"))
        x_axis_layout.addWidget(self.x_combo)

        y_axis_layout = QHBoxLayout()
        self.y_combo = QComboBox()
        self.y_combo.addItems(
            [item for item in self.props.columns if item != "index"]
        )
        y_axis_layout.addWidget(QLabel("y-axis"))
        y_axis_layout.addWidget(self.y_combo)

        self.x_combo.currentIndexChanged.connect(self._update_plot)
        self.y_combo.currentIndexChanged.connect(self._update_plot)

        color_group_layout = QHBoxLayout()
        self.group_combo = QComboBox()
        self.group_combo.addItems(
            [item for item in self.props.columns if item != "index"]
        )
        self.group_combo.currentIndexChanged.connect(self._update_plot)
        color_group_layout.addWidget(QLabel("Group color"))
        color_group_layout.addWidget(self.group_combo)

        dropdown_layout = QVBoxLayout()
        dropdown_layout.addLayout(x_axis_layout)
        dropdown_layout.addLayout(y_axis_layout)
        dropdown_layout.addLayout(color_group_layout)
        dropdown_widget = QWidget()
        dropdown_widget.setLayout(dropdown_layout)

        # Create and apply a horizontal layout for the dropdown widget, toolbar and canvas.
        plotting_layout = QVBoxLayout()
        plotting_layout.addWidget(dropdown_widget)
        plotting_layout.addWidget(self.toolbar)
        plotting_layout.addWidget(self.plot_canvas)
        self.setLayout(plotting_layout)

        # create a colormap
        unique_labels = self.props["label"].unique()
        label_colors = [
            to_rgb(self.labels.get_color(label)) for label in unique_labels
        ]
        self.label_cmap = ListedColormap(label_colors)

        # Update the plot in case the user uses the "Show Selected Labe" optin in the labels menu.
        self.labels.events.show_selected_label.connect(self._update_plot)
        self.labels.events.selected_label.connect(self._update_plot)

        self._update_plot()

    def _update_properties(self, filtered_measurements):
        """Update the properties and regenerate the plot"""
        self.props = filtered_measurements[filtered_measurements["selected"]]
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the plot by plotting the features selected by the user.

        In the case the 'label' column is selected as group, apply the same colors as in the labels layer to the scatter points.
        In the case time is on the x-axis and the 'label' column is selected as group, connect data points with a line.
        In the case the user has clicked the 'Show selected label' option and the group is set to 'label', plot only the points belonging to that label.
        In the case any other feature than 'label' is selected for the group, apply a continuous colormap instead.
        """

        x_axis_property = self.x_combo.currentText()
        y_axis_property = self.y_combo.currentText()
        group = self.group_combo.currentText()

        # Clear data points, and reset the axis scaling and labels.
        for artist in self.ax.lines + self.ax.collections:
            artist.remove()
        self.ax.set_xlabel(x_axis_property)
        self.ax.set_ylabel(y_axis_property)
        self.ax.relim()  # Recalculate limits for the current data
        self.ax.autoscale_view()  # Update the view to include the new limits

        # Check whether to plot with categorical 'label' colors or with a continuous colormap.
        if group == "label":
            # In the case the user wants to see the selected label only, also only plot the selected label. Otherwise, plot all labels
            if self.labels.show_selected_label:
                label = self.labels.selected_label
                plotting_data = self.props[self.props["label"] == label]
                label_colors = [to_rgb(self.labels.get_color(label))]
                colormap = ListedColormap(label_colors)
            else:
                plotting_data = self.props
                colormap = self.label_cmap

            # Group by label and assign the corresponding label colors to the data points
            self.ax.clear()

            self.ax.scatter(
                plotting_data[x_axis_property],
                plotting_data[y_axis_property],
                c=plotting_data["label"],
                cmap=colormap,
                s=10,
            )

            # for label, data in plotting_data.groupby('label'):
            #     label_color = to_rgb(self.labels.get_color(label))
            #     self.ax.scatter(data[x_axis_property], data[y_axis_property], color = label_color, s=10, alpha=1, label=f'Cell {label} ({len(data)} points)')
            #     if x_axis_property == "time_point":
            #         self.ax.plot(data[x_axis_property], data[y_axis_property], color = label_color, alpha=1)

            self.ax.set_ylabel(y_axis_property)
            self.ax.set_xlabel(x_axis_property)
            self.ax.xaxis.label.set_color("white")
            self.ax.yaxis.label.set_color("white")

            if len(plotting_data) < 20:
                self.ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                    framealpha=0.2,
                    edgecolor="white",
                    labelcolor="w",
                    frameon=True,
                )
            else:
                self.ax.legend().remove()

        else:
            # Plot data points on a continuous colormap.
            self.ax.scatter(
                self.props[x_axis_property],
                self.props[y_axis_property],
                c=self.props[group],
                cmap="summer",
                s=10,
            )

        self.plot_canvas.draw()
