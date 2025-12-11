import dask.array as da
import napari
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.utils import CyclicLabelColormap, DirectLabelColormap
from qtpy.QtCore import QEvent, QModelIndex, QObject, Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ClickToSingleSelectFilter(QObject):
    """Event filter to make plain left-clicks act like single selection
    while still allowing Shift/Ctrl clicks to behave normally (append/range)."""

    def __init__(self, table_widget):
        super().__init__(table_widget)
        self.table = table_widget

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            modifiers = event.modifiers()
            if not (
                modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.MetaModifier)
            ):
                self.table.clearSelection()
            return False

        return False


class FloatDelegate(QStyledItemDelegate):
    def __init__(self, decimals, parent=None):
        super().__init__(parent)
        self.nDecimals = decimals

    def displayText(self, value, locale):
        try:
            number = float(value)
        except (ValueError, TypeError):
            return str(value)

        if number.is_integer():
            return str(int(number))
        return f"{number:.{self.nDecimals}f}"


class CustomTableWidget(QTableWidget):
    def mousePressEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid():
            shift = bool(event.modifiers() & Qt.ShiftModifier)
            right = event.button() == Qt.RightButton
            self.parent()._clicked_table(right=right, shift=shift, index=index)

        # Call super so selection behavior still works
        super().mousePressEvent(event)


class ColoredTableWidget(QWidget):
    """Customized table widget with colored rows based on label colors in a napari Labels layer"""

    def __init__(self, layer: "napari.layers.Layer", viewer: "napari.Viewer" = None):
        super().__init__()

        self._layer = layer
        self._viewer = viewer
        self._table_widget = CustomTableWidget()
        self.special_selection = []

        self._layer.events.colormap.connect(self._set_label_colors_to_rows)
        self._layer.events.show_selected_label.connect(self._set_label_colors_to_rows)
        if hasattr(layer, "properties"):
            self._set_data(layer.properties)
        else:
            self._set_data({})
        self.ascending = False  # for choosing whether to sort ascending or descending

        # Connect to single click in the header to sort the table.
        self._table_widget.horizontalHeader().sectionClicked.connect(self._sort_table)

        # Instruction label to explain left and right mouse click.
        label = QLabel(
            "Use left mouse click to select and center a label, use right mouse click to show the selected label only. Use SHIFT for multi-selection."
        )
        label.setWordWrap(True)
        font = label.font()
        font.setItalic(True)
        label.setFont(font)

        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_table)

        save_button = QPushButton("Save as csv")
        save_button.clicked.connect(self._save_table)

        button_layout = QHBoxLayout()
        button_layout.addWidget(copy_button)
        button_layout.addWidget(save_button)

        delete_undo_layout = QHBoxLayout()
        delete_button = QPushButton("Delete selected labels")
        delete_button.clicked.connect(self._delete_labels)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._undo_delete)
        delete_undo_layout.addWidget(delete_button)
        delete_undo_layout.addWidget(self.undo_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(delete_undo_layout)
        main_layout.addWidget(label)
        main_layout.addWidget(self._table_widget)
        self.setLayout(main_layout)
        self.setMinimumHeight(300)

        self._table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self._table_widget.setStyleSheet("""
            QTableWidget::item:selected {
                border: 2px solid cyan;
            }
        """)

        self._table_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self._table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)

        self._click_filter = ClickToSingleSelectFilter(self._table_widget)
        self._table_widget.viewport().installEventFilter(self._click_filter)

    def select_label(
        self, position: list[int | float], label: int, append: bool = False
    ):
        """Select the label that was clicked on."""

        if label is None or label == 0:
            self._table_widget.clearSelection()
            self._reset_layer_colormap()
            return

        if "time_point" in self._table:
            time_dim = self._layer.metadata["dimensions"].index("T")
            t = position[time_dim]
            row = self._find_row(time_point=t, label=label)
            self._select_row(row, append)

        else:
            row = self._find_row(label=label)
            self._select_row(row, append)

        self._update_label_colormap()

    def _select_row(self, row: int, append: bool):
        """Select a row visually in the table."""

        if row is None:
            return
        if not append:
            self._table_widget.clearSelection()
        self._table_widget.selectRow(row)
        self._table_widget.scrollToItem(self._table_widget.item(row, 0))

    def _find_row(self, **conditions):
        """
        Find the first row matching the given conditions (eg label=12, time_point=5)
        Returns: row index or None
        """

        n_rows = self._table_widget.rowCount()

        for row in range(n_rows):
            if all(
                float(self._table[col][row]) == float(val)
                for col, val in conditions.items()
            ):
                return row

        return None

    def _delete_labels(self):
        """Delete the selected labels in the table and store state for undo"""

        selected_rows = sorted(
            {index.row() for index in self._table_widget.selectedIndexes()}
        )
        if not selected_rows:
            return

        self._undo_info = []

        # Delete labels from the table itself
        for row in reversed(selected_rows):
            row_data = {col: self._table[col][row] for col in self._table}
            self._undo_info.append({"row": row, "row_data": row_data})
            for col in self._table:
                self._table[col] = np.delete(self._table[col], row, axis=0)
            self._table_widget.removeRow(row)

        # Delete from layer.data
        if "time_point" in self._table:
            time_dim = self._layer.metadata["dimensions"].index("T")
            for info in self._undo_info:
                row_data = info["row_data"]
                t = row_data["time_point"]
                label = row_data["label"]

                # Build slice for this time point
                sl = [slice(None)] * self._layer.data.ndim
                sl[time_dim] = int(t)

                # Extract the slice and store previous state
                sliced_data = self._layer.data[tuple(sl)]
                if isinstance(sliced_data, da.core.Array):
                    sliced_data = sliced_data.compute()
                # store only the boolean mask positions affected by the label
                mask = sliced_data == int(label)
                prev_values = sliced_data[mask].copy()

                info["slice"] = sl
                info["mask"] = mask
                info["prev_values"] = prev_values

                # Set label to 0
                sliced_data[mask] = 0
                # assign back to layer
                self._layer.data[tuple(sl)] = sliced_data

        else:
            # no time_point → delete across full volume
            for info in self._undo_info:
                label = info["row_data"]["label"]
                mask = self._layer.data == int(label)
                prev_values = self._layer.data[mask].copy()
                info["mask"] = mask
                info["prev_values"] = prev_values
                self._layer.data[mask] = 0

        # Enable undo button
        self.undo_button.setEnabled(True)
        self._layer.data = self._layer.data

    def _undo_delete(self):
        """Restore previously deleted labels and table rows"""

        if not hasattr(self, "_undo_info") or not self._undo_info:
            return

        # Sort by row ascending so insert indices stay correct
        for info in sorted(self._undo_info, key=lambda x: x["row"]):
            row = info["row"]
            row_data = info["row_data"]

            # Restore table data (NumPy arrays)
            for col in self._table:
                # Make sure to wrap scalar as list for np.insert
                value_to_insert = np.array([row_data[col]])
                self._table[col] = np.insert(
                    self._table[col], row, value_to_insert, axis=0
                )

            # Restore QTableWidget row visually
            self._table_widget.insertRow(row)
            for col_idx, col in enumerate(self._table):
                item = QTableWidgetItem(str(row_data[col]))
                self._table_widget.setItem(row, col_idx, item)

            # Restore layer.data for this slice
            sl = info.get("slice")
            if sl is not None:
                sliced_data = self._layer.data[tuple(sl)]
                if isinstance(sliced_data, da.core.Array):
                    sliced_data = sliced_data.compute()
                sliced_data[info["mask"]] = info["prev_values"]
                self._layer.data[tuple(sl)] = sliced_data
            else:
                # global volume undo
                self._layer.data[info["mask"]] = info["prev_values"]

        # Clear undo info and disable button
        self._undo_info = []
        self.undo_button.setEnabled(False)

        # refresh
        self._layer.data = self._layer.data

    def _set_data(self, table: dict) -> None:
        """Set the content of the table from a dictionary"""

        self._table = table
        self._layer.properties = table

        self._table_widget.clear()
        try:
            self._table_widget.setRowCount(len(next(iter(table.values()))))
            self._table_widget.setColumnCount(len(table))
        except StopIteration:
            pass

        for i, column in enumerate(table):
            self._table_widget.setHorizontalHeaderItem(i, QTableWidgetItem(column))
            for j, value in enumerate(table.get(column)):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table_widget.setItem(j, i, item)

        self._table_widget.setItemDelegate(FloatDelegate(3, self._table_widget))

        self._set_label_colors_to_rows()

    def _set_label_colors_to_rows(self) -> None:
        """Apply the colors of the napari label image to the table"""

        for i in range(self._table_widget.rowCount()):
            label = self._table["label"][i]
            label_color = to_rgba(self._layer.colormap.map(label))

            if label_color[3] == 0:
                label_color = [0, 0, 0, 0]

            r, g, b = (
                int(label_color[0] * 255),
                int(label_color[1] * 255),
                int(label_color[2] * 255),
            )

            qcolor = QColor(r, g, b)

            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = QColor(0, 0, 0) if luminance > 140 else QColor(255, 255, 255)

            for j in range(self._table_widget.columnCount()):
                item = self._table_widget.item(i, j)
                item.setBackground(qcolor)
                item.setForeground(text_color)

    def _save_table(self) -> None:
        """Save table to csv file"""
        filename, _ = QFileDialog.getSaveFileName(self, "Save as csv", ".", "*.csv")
        pd.DataFrame(self._table).to_csv(filename)

    def _copy_table(self) -> None:
        """Copy table to clipboard"""
        pd.DataFrame(self._table).to_clipboard()

    def _clicked_table(self, right: bool, shift: bool, index: QModelIndex) -> None:
        """Center the viewer to clicked label. If the right mouse button was used, set
        layer.show_selected_label to True, else False.
        """
        row = index.row()
        label = self._table["label"][row]
        self._layer.selected_label = label
        spatial_columns = sorted([key for key in self._table if "centroid" in key])
        spatial_coords = [int(self._table[col][row]) for col in spatial_columns]

        if "dimensions" in self._layer.metadata:
            dims = self._layer.metadata["dimensions"]
        else:
            ndim = self._layer.data.ndim
            dims = ["C", "T", "Z", "Y", "X"][:-ndim]

        step = list(self._viewer.dims.current_step)
        if "time_point" in self._table:
            step[dims.index("T")] = int(self._table["time_point"][row])
        if len(spatial_coords) == 3:
            step[dims.index("Z")] = spatial_coords[-3]
        step[-2] = spatial_coords[-2]
        step[-1] = spatial_coords[-1]

        self._viewer.dims.current_step = step

        if right:
            if not shift:
                self.special_selection = [label]
            else:
                self.special_selection.append(label)
            self._table_widget.clearSelection()
        else:
            self.special_selection = []
        QTimer.singleShot(0, self._update_label_colormap)

    def _update_label_colormap(self):
        """
        Highlight the labels of selected rows.
        """

        # replace cyclic map by direct map if necessary
        if isinstance(self._layer.colormap, CyclicLabelColormap):
            labels = self._table["label"]
            colors = [self._layer.colormap.map(label) for label in labels]
            self._layer.colormap = DirectLabelColormap(
                color_dict={
                    **dict(zip(labels, colors, strict=True)),
                    None: [0, 0, 0, 0],
                }
            )

        # in case of right-click on the table, we should only show the selected label(s)
        if len(self.special_selection) != 0:
            for _, color in self._layer.colormap.color_dict.items():
                color[-1] = 0
            for key in self.special_selection:
                if key in self._layer.colormap.color_dict:
                    self._layer.colormap.color_dict[key][-1] = 1

        # find selected rows, and set highlight matching labels
        else:
            selected_rows = sorted(
                {index.row() for index in self._table_widget.selectedIndexes()}
            )
            if not selected_rows:
                self._reset_layer_colormap()
                return

            selected_labels = [self._table["label"][row] for row in selected_rows]
            for key, color in self._layer.colormap.color_dict.items():
                if key is not None and key != 0:
                    color[-1] = 0.6
            for key in selected_labels:
                if key in self._layer.colormap.color_dict:
                    self._layer.colormap.color_dict[key][-1] = 1

        self._layer.colormap = DirectLabelColormap(
            color_dict=self._layer.colormap.color_dict
        )

    def _reset_layer_colormap(self):
        """Set all alpha values back to 1 to reset the colormap"""

        self.special_selection = []
        if self._layer is not None:
            for key in self._layer.colormap.color_dict:
                if key is not None and key != 0:
                    self._layer.colormap.color_dict[key][-1] = 1
            self._layer.colormap = DirectLabelColormap(
                color_dict=self._layer.colormap.color_dict
            )

    def _sort_table(self) -> None:
        """Sorts the table in ascending or descending order"""

        selected_column = list(self._table.keys())[self._table_widget.currentColumn()]
        df = pd.DataFrame(self._table).sort_values(
            by=selected_column, ascending=self.ascending
        )
        self.ascending = not self.ascending

        self._set_data(df.to_dict(orient="list"))
        self._set_label_colors_to_rows()
