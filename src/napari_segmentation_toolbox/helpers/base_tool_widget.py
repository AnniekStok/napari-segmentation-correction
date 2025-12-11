import napari
from psygnal import Signal
from qtpy.QtWidgets import (
    QWidget,
)


class BaseToolWidget(QWidget):
    """Base widget that listens to the napari layer list and activates/deactivates its
    state depending on whether the current active layer is a suitable target for its
    function."""

    update_status = Signal()

    def __init__(self, viewer: "napari.viewer.Viewer", layer_type: tuple) -> None:
        super().__init__()

        self.viewer = viewer
        self.layer_type = layer_type
        self.layer = None

        self.viewer.layers.selection.events.active.connect(self._on_selection_changed)

    def _on_selection_changed(self):
        """Listen to the napari layer list updates, and send a signal to notify the parent
        widget if a single layer is selected that is of the correct type, or if this is
        not the case and the widget should inactivate its button(s)."""

        if (
            len(self.viewer.layers.selection) == 1
        ):  # Only consider single layer selection
            selected_layer = self.viewer.layers.selection.active
            if isinstance(selected_layer, self.layer_type):
                self.layer = selected_layer
            else:
                self.layer = None

            self.update_status.emit()
