import contextlib

import napari
from psygnal import Signal
from qtpy.QtWidgets import QComboBox


class LayerDropdown(QComboBox):
    """QComboBox widget with functions for updating the selected layer and to update the
    list of options when the list of layers is modified."""

    layer_changed = Signal(str)

    def __init__(self, viewer: napari.Viewer, layer_type: tuple, allow_none=False):
        super().__init__()

        self.viewer = viewer
        self.layer_type = layer_type
        self.allow_none = allow_none
        self.selected_layer = None
        self._deleted = False

        # track rename callbacks so we can disconnect them
        self._rename_callbacks = {}

        # viewer connections
        self.viewer.layers.events.inserted.connect(self._on_insert)
        self.viewer.layers.events.changed.connect(self._update_dropdown)
        self.viewer.layers.events.removed.connect(self._on_removed)
        self.viewer.layers.selection.events.changed.connect(self._on_selection_changed)

        self.currentTextChanged.connect(self._emit_layer_changed)

        self._update_dropdown()

    def _on_insert(self, event) -> None:
        """Update dropdown and make new layer responsive to name changes"""

        layer = event.value
        if isinstance(layer, self.layer_type):

            def _rename_cb(evt):
                if not self._deleted:
                    self._update_dropdown()

            layer.events.name.connect(_rename_cb)
            self._rename_callbacks[layer] = _rename_cb

            self._update_dropdown()

    def _on_removed(self, event) -> None:
        """Disconnect signals and update dropdown when a layer is removed."""

        if self._deleted:
            return

        layer = event.value
        # disconnect rename callback if present
        cb = self._rename_callbacks.pop(layer, None)
        if cb:
            with contextlib.suppress(Exception):
                layer.events.name.disconnect(cb)

        self._update_dropdown()

    def _on_selection_changed(self):
        """Update the active layer when the selection changes"""
        if self._deleted:
            return

        if len(self.viewer.layers.selection) == 1:
            selected = self.viewer.layers.selection.active
            if (
                isinstance(selected, self.layer_type)
                and selected != self.selected_layer
            ):
                self.setCurrentText(selected.name)
                self._emit_layer_changed()

    def _update_dropdown(self, event=None) -> None:
        """Update the layers in the dropdown"""

        if self._deleted:
            return

        previous = self.currentText()
        self.clear()

        layers = [
            layer for layer in self.viewer.layers if isinstance(layer, self.layer_type)
        ]

        items = []

        if self.allow_none:
            self.addItem("No selection")
            items.append("No selection")

        for layer in layers:
            self.addItem(layer.name)
            items.append(layer.name)

        # restore selection if still valid
        if previous in items:
            self.setCurrentText(previous)

    def _emit_layer_changed(self) -> None:
        """Emit a signal holding the currently selected layer"""

        if self._deleted:
            return

        name = self.currentText()

        if name != "No selection" and name in self.viewer.layers:
            self.selected_layer = self.viewer.layers[name]
        else:
            self.selected_layer = None
            name = ""

        self.layer_changed.emit(name)

    def deleteLater(self) -> None:
        """Disconnect everything cleanly."""

        with contextlib.suppress(Exception):
            self.viewer.layers.events.inserted.disconnect(self._on_insert)
        with contextlib.suppress(Exception):
            self.viewer.layers.events.changed.disconnect(self._update_dropdown)
        with contextlib.suppress(Exception):
            self.viewer.layers.events.removed.disconnect(self._on_removed)
        with contextlib.suppress(Exception):
            self.viewer.layers.selection.events.changed.disconnect(
                self._on_selection_changed
            )

        # per-layer callbacks
        for layer, cb in list(self._rename_callbacks.items()):
            with contextlib.suppress(Exception):
                layer.events.name.disconnect(cb)
        self._rename_callbacks.clear()

        # qt signal
        with contextlib.suppress(Exception):
            self.currentTextChanged.disconnect(self._emit_layer_changed)

        self._deleted = True
        super().deleteLater()
