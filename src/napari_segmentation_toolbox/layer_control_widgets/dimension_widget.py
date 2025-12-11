import napari
import numpy as np
from psygnal import Signal
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_segmentation_toolbox.helpers.base_tool_widget import BaseToolWidget


class DimensionWidget(BaseToolWidget):
    """Unified widget for axis order, axis names and voxel scaling.

    Layout: 3 columns (Axis order | Names | Scaling) with headers.
    - Axis column: shows original dim index + size and a combobox to pick a new axis position,
      plus an "Apply" button for reordering.
    - Name column: combobox to choose a name (C/T/Z) for non-Y/X dims, and Y/X labels for last two dims.
      Has an "Apply" button which writes names to layer.metadata['dimensions'].
    - Scale column: QDoubleSpinBox per axis. Changes are applied immediately (no apply button).
    """

    update_dims = Signal()

    EXTRA_AXIS_NAMES = ["C", "T", "Z"]

    def __init__(
        self,
        viewer: napari.Viewer,
        layer_type=(napari.layers.Labels, napari.layers.Image),
    ):
        super().__init__(viewer, layer_type)

        self.max_dims = 5  # C,T,Z,Y,X

        group = QGroupBox("Dimensions")
        vbox = QVBoxLayout()
        self.grid = QGridLayout()

        # Headers
        self.grid.addWidget(QLabel("Axis"), 0, 0, alignment=Qt.AlignHCenter)
        self.grid.addWidget(QLabel("New axis order"), 0, 1, alignment=Qt.AlignHCenter)
        self.grid.addWidget(QLabel("Name"), 0, 2, alignment=Qt.AlignHCenter)
        self.grid.addWidget(QLabel("Pixel scaling"), 0, 3, alignment=Qt.AlignHCenter)

        # Per-row widgets
        self.orig_labels = []  # QLabel showing e.g. "0 [512]"
        self.pos_combos = []  # combobox to pick new position (for reordering)
        self.name_widgets = []  # combobox for C/T/Z, disabled for Y/X
        self.scale_widgets = []  # QDoubleSpinBox for scale

        for i in range(self.max_dims):
            lbl = QLabel("")  # will show "i [shape]"
            lbl.setVisible(False)
            self.grid.addWidget(lbl, i + 1, 0)
            self.orig_labels.append(lbl)

            pos_combo = QComboBox()
            pos_combo.setVisible(False)
            pos_combo.setEnabled(False)
            pos_combo.currentIndexChanged.connect(self._update_axis_apply_state)
            self.grid.addWidget(
                pos_combo, i + 1, 1, alignment=Qt.AlignRight
            )  # position combobox to the right of label
            self.pos_combos.append(pos_combo)

            name_widget = QComboBox()
            name_widget.setVisible(False)
            name_widget.setEnabled(False)
            name_widget.currentIndexChanged.connect(self._update_name_apply_state)

            self.grid.addWidget(name_widget, i + 1, 2)
            self.name_widgets.append(name_widget)

            scale_spin = QDoubleSpinBox()
            scale_spin.setSingleStep(0.1)
            scale_spin.setMinimum(0.01)
            scale_spin.setValue(1.0)
            scale_spin.setVisible(False)
            scale_spin.setEnabled(False)
            scale_spin.editingFinished.connect(
                self._apply_scale_single
            )  # apply directly
            self.grid.addWidget(scale_spin, i + 1, 3)
            self.scale_widgets.append(scale_spin)

        # Buttons row (under each column)
        self.apply_axis_btn = QPushButton("Apply")
        self.apply_axis_btn.setEnabled(False)
        self.apply_axis_btn.clicked.connect(self._apply_axis_reorder)
        self.grid.addWidget(
            self.apply_axis_btn, self.max_dims + 1, 1, alignment=Qt.AlignCenter
        )

        self.apply_name_btn = QPushButton("Apply")
        self.apply_name_btn.setEnabled(False)
        self.apply_name_btn.clicked.connect(self._apply_names)
        self.grid.addWidget(
            self.apply_name_btn, self.max_dims + 1, 2, alignment=Qt.AlignCenter
        )

        # no apply for scale; changes apply immediately
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.grid.addWidget(spacer, self.max_dims + 1, 2)

        group.setLayout(self.grid)
        vbox.addWidget(group)
        self.setLayout(vbox)

        self.grid.setColumnStretch(0, 0)  # Axis (label)
        self.grid.setColumnStretch(1, 0)  # New axis order combobox
        self.grid.setColumnStretch(2, 1)  # Name (gets 50%)
        self.grid.setColumnStretch(3, 1)  # Pixel scaling (gets 50%)

        # connect viewer/layer changes
        self.update_status.connect(self._update_from_layer)

    def _update_from_layer(self) -> None:
        """Populate widgets to reflect the current layer state."""

        if self.layer is None:
            # hide everything if no layer
            for lst in (
                self.orig_labels,
                self.pos_combos,
                self.name_widgets,
                self.scale_widgets,
            ):
                for w in lst:
                    w.setVisible(False)
                    w.setEnabled(False)
            self.apply_axis_btn.setEnabled(False)
            self.apply_name_btn.setEnabled(False)
            return

        ndim = self.layer.data.ndim
        shape = self.layer.data.shape
        scale_info = getattr(self.layer, "scale", tuple([1] * ndim))

        # Build list of valid positions for reordering (0..ndim-1)
        positions = [str(i) for i in range(ndim)]

        offset = self.max_dims - ndim
        # Fill rows
        for i in range(self.max_dims):
            if i < ndim:
                # orig label
                self.orig_labels[i].setVisible(True)
                self.orig_labels[i].setEnabled(True)
                self.orig_labels[i].setText(f"{i} [{shape[i]}]")

                # position combobox: show indices 0..ndim-1, default to current i
                combo = self.pos_combos[i]
                with QSignalBlocker(combo):
                    combo.setVisible(True)
                    combo.setEnabled(True)
                    combo.clear()
                    combo.addItems(positions)
                    combo.setCurrentIndex(i)

                # name widget:
                name_w = self.name_widgets[i]
                name_w.setVisible(True)
                name_w.setEnabled(True)
                with QSignalBlocker(name_w):
                    if i >= ndim - 2:
                        # last two are Y, X — show a disabled combobox
                        name_w.clear()
                        name_w.addItems(["Y", "X"] if (ndim - i == 2) else ["X"])
                        name_w.setEnabled(False)
                        name_w.setCurrentIndex(0)
                    else:
                        # non-Y/X axes: allow EXTRA_AXIS_NAMES
                        name_w.clear()
                        name_w.addItems(self.EXTRA_AXIS_NAMES)
                        name_w.setEnabled(True)
                        name_w.setCurrentIndex(i + offset)

                # scale
                scale_w = self.scale_widgets[i]
                scale_w.setVisible(True)
                scale_w.setEnabled(True)
                # set value (block signals while setting)
                with QSignalBlocker(scale_w):
                    val = scale_info[i] if i < len(scale_info) else 1.0
                    scale_w.setValue(float(val))

            else:
                # hide unused rows
                for w in (
                    self.orig_labels[i],
                    self.pos_combos[i],
                    self.name_widgets[i],
                    self.scale_widgets[i],
                ):
                    w.setVisible(False)
                    w.setEnabled(False)

        # Populate name comboboxes from layer.metadata['dimensions'] if present
        if "dimensions" in self.layer.metadata:
            stored = list(self.layer.metadata.get("dimensions", []))
            # If stored length mismatches, fill in from scratch
            if len(stored) != ndim:
                stored = self.EXTRA_AXIS_NAMES[offset:] + ["Y", "X"]

            # Set comboboxes according to stored values
            for i in range(ndim):
                w = self.name_widgets[i]
                idx = w.findText(stored[i])
                if idx != -1:
                    w.setCurrentIndex(idx)
                else:
                    w.setCurrentIndex(min(i, len(self.EXTRA_AXIS_NAMES) - 1))

        # Refresh the apply button states
        self._update_axis_apply_state()

        # immediately apply the current dimension names but do not emit signal
        self.layer.metadata["dimensions"] = [
            self.name_widgets[i].currentText() for i in range(ndim)
        ]
        self._update_name_apply_state()

    def _update_axis_apply_state(self) -> None:
        """Enable apply button only if a valid, different ordering has been selected."""

        if self.layer is None:
            self.apply_axis_btn.setEnabled(False)
            return

        ndim = self.layer.data.ndim

        # read chosen positions for enabled combos
        chosen = [int(self.pos_combos[i].currentText()) for i in range(ndim)]

        # if duplicates or identical to identity, disable
        if len(chosen) != len(set(chosen)) or chosen == list(range(ndim)):
            self.apply_axis_btn.setEnabled(False)
            self.apply_axis_btn.setStyleSheet("")
        else:
            self.apply_axis_btn.setEnabled(True)
            self.apply_axis_btn.setStyleSheet("QPushButton { border: 2px solid cyan; }")

    def _update_name_apply_state(self) -> None:
        """Enable apply for name only when names are valid and different from current metadata (if present)."""

        if self.layer is None:
            self.apply_name_btn.setEnabled(False)
            return

        names = [w.currentText() for w in self.name_widgets if w.isEnabled()] + [
            "Y",
            "X",
        ]
        stored = self.layer.metadata.get("dimensions", None)

        # disable in case of duplicates, or if the current names are the same as the stored ones
        if len(names) != len(set(names)) or names == stored:
            self.apply_name_btn.setEnabled(False)
            self.apply_name_btn.setStyleSheet("")
            return

        self.apply_name_btn.setEnabled(True)
        self.apply_name_btn.setStyleSheet("QPushButton { border: 2px solid cyan; }")

    def _apply_axis_reorder(self):
        """Read position comboboxes and transpose layer.data accordingly."""

        if self.layer is None:
            return

        ndim = self.layer.data.ndim
        new_order = [int(c.currentText()) for c in self.pos_combos if c.isEnabled()]

        if new_order == list(range(ndim)) or len(new_order) != len(set(new_order)):
            return

        # transpose data
        self.layer.data = np.transpose(self.layer.data, new_order)

        # after transposing, reset viewer view update status
        old_step = self.viewer.dims.current_step
        self.viewer.reset_view()
        self.viewer.dims.current_step = old_step
        self.update_status.emit()
        self.apply_axis_btn.setStyleSheet("")
        self.apply_axis_btn.setEnabled(False)

    def _apply_names(self):
        """Write dimension names into metadata and notify."""

        if self.layer is None:
            return

        ndim = self.layer.data.ndim
        names = [self.name_widgets[i].currentText() for i in range(ndim)]
        if len(names) != len(set(names)):
            return
        self.layer.metadata["dimensions"] = names
        self.update_status.emit()
        self.apply_name_btn.setStyleSheet("")
        self.apply_name_btn.setEnabled(False)

    def _apply_scale_single(self):
        """Apply current scale values for enabled spinboxes to the layer. This is immediate (no apply button)."""

        if self.layer is None:
            return

        scale_values = [w.value() for w in self.scale_widgets if w.isEnabled()]

        old_step = self.viewer.dims.current_step
        self.layer.scale = tuple(scale_values)
        self.viewer.reset_view()
        self.viewer.dims.current_step = old_step
        self.update_status.emit()
