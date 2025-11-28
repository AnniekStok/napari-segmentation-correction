import napari
import pandas as pd
from matplotlib.colors import to_rgb
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
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
            right = event.button() == Qt.RightButton
            self.parent()._clicked_table(right=right, index=index)

        # Call super so selection behavior still works
        super().mousePressEvent(event)


class ColoredTableWidget(QWidget):
    """Customized table widget with colored rows based on label colors in a napari Labels layer"""

    def __init__(self, layer: "napari.layers.Layer", viewer: "napari.Viewer" = None):
        super().__init__()

        self._layer = layer
        self._viewer = viewer
        self._table_widget = CustomTableWidget()

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
            "Use left mouse click to select and center a label, use right mouse click to show the selected label only."
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
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(label)
        main_layout.addWidget(self._table_widget)
        self.setLayout(main_layout)
        self.setMinimumHeight(300)

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
                self._table_widget.setItem(j, i, QTableWidgetItem(str(value)))

        self._table_widget.setItemDelegate(FloatDelegate(3, self._table_widget))

        self._set_label_colors_to_rows()

    def _set_label_colors_to_rows(self) -> None:
        """Apply the colors of the napari label image to the table"""

        for i in range(self._table_widget.rowCount()):
            label = self._table["label"][i]
            label_color = to_rgb(self._layer.colormap.map(label))
            scaled_color = (
                int(label_color[0] * 255),
                int(label_color[1] * 255),
                int(label_color[2] * 255),
            )
            for j in range(self._table_widget.columnCount()):
                self._table_widget.item(i, j).setBackground(QColor(*scaled_color))

    def _save_table(self) -> None:
        """Save table to csv file"""
        filename, _ = QFileDialog.getSaveFileName(self, "Save as csv", ".", "*.csv")
        pd.DataFrame(self._table).to_csv(filename)

    def _copy_table(self) -> None:
        """Copy table to clipboard"""
        pd.DataFrame(self._table).to_clipboard()

    def _clicked_table(self, right: bool, index: QModelIndex) -> None:
        """Center the viewer to clicked label. If the right mouse button was used, set
        layer.show_selected_label to True, else False.
        """
        row = index.row()
        label = self._table["label"][row]
        self._layer.selected_label = label
        self._layer.show_selected_label = right
        spatial_columns = sorted([key for key in self._table if "centroid" in key])
        spatial_coords = [int(self._table[col][row]) for col in spatial_columns]

        if "time_point" in self._table:
            new_step = (int(self._table["time_point"][row]), *spatial_coords)
        else:
            new_step = spatial_coords

        self._viewer.dims.current_step = new_step

    def _sort_table(self) -> None:
        """Sorts the table in ascending or descending order"""

        selected_column = list(self._table.keys())[self._table_widget.currentColumn()]
        df = pd.DataFrame(self._table).sort_values(
            by=selected_column, ascending=self.ascending
        )
        self.ascending = not self.ascending

        self._set_data(df.to_dict(orient="list"))
        self._set_label_colors_to_rows()
