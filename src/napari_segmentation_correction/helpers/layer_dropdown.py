import contextlib
import weakref

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
        self._rename_callbacks: dict[int, tuple[weakref.ref, callable]] = {}

        self.destroyed.connect(self._on_destroyed)

        # viewer connections
        self.viewer.layers.events.inserted.connect(self._on_insert)
        self.viewer.layers.events.changed.connect(self._update_dropdown)
        self.viewer.layers.events.removed.connect(self._on_removed)
        self.viewer.layers.selection.events.changed.connect(self._on_selection_changed)

        self.currentTextChanged.connect(self._emit_layer_changed)
        self._update_dropdown()

    def _make_weak_rename_cb(self, layer):
        self_ref = weakref.ref(self)

        def _rename_cb(event=None):
            self_obj = self_ref()
            if self_obj is None or self_obj._deleted:
                return
            with contextlib.suppress(AttributeError, RuntimeError):
                self_obj._update_dropdown()

        return _rename_cb

    def _on_insert(self, event) -> None:
        """Update dropdown and make new layer responsive to name changes"""

        if self._deleted:
            return

        layer = event.value
        if isinstance(layer, self.layer_type):
            cb = self._make_weak_rename_cb(layer)
            layer.events.name.connect(cb)
            self._rename_callbacks[id(layer)] = (weakref.ref(layer), cb)
            self._update_dropdown()

    def _on_removed(self, event) -> None:
        """Disconnect signals and update dropdown when a layer is removed."""

        if self._deleted:
            return

        layer = event.value
        pair = self._rename_callbacks.pop(id(layer), None)
        if pair is not None:
            layer_ref, cb = pair
            layer_obj = layer_ref() if layer_ref else None
            target = layer_obj if layer_obj else layer
            with contextlib.suppress(AttributeError, RuntimeError, TypeError):
                target.events.name.disconnect(cb)

        self._update_dropdown()

    def _on_selection_changed(self):
        """Update the active layer when the selection changes"""
        if self._deleted:
            return

        try:
            if len(self.viewer.layers.selection) == 1:
                selected = self.viewer.layers.selection.active
                if (
                    isinstance(selected, self.layer_type)
                    and selected != self.selected_layer
                ):
                    self.setCurrentText(selected.name)
                    self._emit_layer_changed()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _update_dropdown(self, event=None) -> None:
        """Update the layers in the dropdown"""

        if self._deleted:
            return

        try:
            previous = self.currentText()
            self.clear()

            layers = [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, self.layer_type)
            ]

            names = []
            if self.allow_none:
                self.addItem("No selection")
                names.append("No selection")

            for layer in layers:
                self.addItem(layer.name)
                names.append(layer.name)

            # restore previous selection if still valid
            if previous in names:
                self.setCurrentText(previous)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _emit_layer_changed(self) -> None:
        """Emit a signal holding the currently selected layer"""

        if self._deleted:
            return

        try:
            name = self.currentText()
            if name != "No selection" and name in self.viewer.layers:
                self.selected_layer = self.viewer.layers[name]
            else:
                self.selected_layer = None
                name = ""
            self.layer_changed.emit(name)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _on_destroyed(self, *args):
        """Disconnect everything cleanly"""

        self._deleted = True

        with contextlib.suppress(AttributeError, RuntimeError, TypeError):
            self.viewer.layers.events.inserted.disconnect(self._on_insert)
            self.viewer.layers.events.changed.disconnect(self._update_dropdown)
            self.viewer.layers.events.removed.disconnect(self._on_removed)
            self.viewer.layers.selection.events.changed.disconnect(
                self._on_selection_changed
            )

        for layer_ref, cb in self._rename_callbacks.values():
            layer_obj = layer_ref() if layer_ref else None
            target = layer_obj
            if target:
                with contextlib.suppress(AttributeError, RuntimeError, TypeError):
                    target.events.name.disconnect(cb)

        self._rename_callbacks.clear()

        with contextlib.suppress(AttributeError, RuntimeError, TypeError):
            self.currentTextChanged.disconnect(self._emit_layer_changed)
