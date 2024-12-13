import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.colors import ListedColormap, to_rgb
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .layer_manager import LayerManager

ICON_ROOT = Path(__file__).parent / "icons"


class PlotWidget(QWidget):
    """Matplotlib widget that displays features of the selected labels layer
    """

    def __init__(self, label_manager: LayerManager):
        super().__init__()

        self.label_manager = label_manager
        self.label_manager.layer_update.connect(self._layer_update)

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
        x_axis_layout.addWidget(QLabel("x-axis"))
        x_axis_layout.addWidget(self.x_combo)

        y_axis_layout = QHBoxLayout()
        self.y_combo = QComboBox()
        y_axis_layout.addWidget(QLabel("y-axis"))
        y_axis_layout.addWidget(self.y_combo)

        self.x_combo.currentIndexChanged.connect(self._update_plot)
        self.y_combo.currentIndexChanged.connect(self._update_plot)

        color_group_layout = QHBoxLayout()
        self.group_combo = QComboBox()
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

    def _layer_update(self) -> None:
        """Connect events to plot updates"""

        if self.label_manager.selected_layer is not None:
            self.label_manager.selected_layer.events.show_selected_label.connect(self._update_plot)
            self.label_manager.selected_layer.events.selected_label.connect(self._update_plot)
            self.label_manager.selected_layer.events.features.connect(self._update_dropdown)

    def _update_dropdown(self) -> None:
        """Update the dropdowns with the column headers"""

        if len(self.label_manager.selected_layer.features) > 0:
            if self.x_combo.count() > 0:
                prev_index = self.x_combo.currentIndex()
            else:
                prev_index = 0
            self.x_combo.clear()
            self.x_combo.addItems(
                [item for item in self.label_manager.selected_layer.features.columns if item != "index"]
            )
            self.x_combo.setCurrentIndex(prev_index)

            if self.y_combo.count() > 0:
                prev_index = self.y_combo.currentIndex()
            else:
                prev_index = 1
            self.y_combo.clear()
            self.y_combo.addItems(
                [item for item in self.label_manager.selected_layer.features.columns if item != "index"]
            )
            self.y_combo.setCurrentIndex(prev_index)

            if self.group_combo.count() > 0:
                prev_index = self.group_combo.currentIndex()
            else:
                prev_index = 0
            self.group_combo.clear()
            self.group_combo.addItems(
                [item for item in self.label_manager.selected_layer.features.columns if item != "index"]
            )
            self.group_combo.setCurrentIndex(prev_index)

    def _update_plot(self) -> None:
        """Update the plot by plotting the features selected by the user."""

        if len(self.label_manager.selected_layer.features) > 0:

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

            if group == "label":
                if self.label_manager.selected_layer.show_selected_label:
                    label = self.label_manager.selected_layer.selected_label
                    plotting_data = self.label_manager.selected_layer.features[
                        self.label_manager.selected_layer.features["label"] == label
                    ]
                    unique_labels = [label]
                else:
                    plotting_data = self.label_manager.selected_layer.features
                    unique_labels = plotting_data["label"].unique()

                # Create consistent label-to-color mapping
                colormap = self.label_manager.selected_layer.colormap
                label_colors = {
                    label: to_rgb(colormap.map(label))
                    for label in unique_labels
                }

                # Scatter plot
                self.ax.scatter(
                    plotting_data[x_axis_property],
                    plotting_data[y_axis_property],
                    c=plotting_data["label"].map(label_colors),
                    cmap=ListedColormap(list(label_colors.values())),  # Consistent colormap
                    s=10,
                )

                # Line plot for time_point x-axis
                if x_axis_property == "time_point":
                    for label, color in label_colors.items():
                        label_data = plotting_data[plotting_data["label"] == label].sort_values(by=x_axis_property)
                        self.ax.plot(
                            label_data[x_axis_property],
                            label_data[y_axis_property],
                            linestyle='-',
                            color=color,
                            linewidth=1,
                        )

            else:
                # Continuous colormap for other grouping
                self.ax.scatter(
                    self.label_manager.selected_layer.features[x_axis_property],
                    self.label_manager.selected_layer.features[y_axis_property],
                    c=self.label_manager.selected_layer.features[group],
                    cmap="summer",
                    s=10,
                )

            self.plot_canvas.draw()

