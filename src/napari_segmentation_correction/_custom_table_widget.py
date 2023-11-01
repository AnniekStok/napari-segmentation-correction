import pandas as pd
from napari_skimage_regionprops import TableWidget
from matplotlib.colors                  import to_rgb
from qtpy.QtCore import Qt

from qtpy.QtGui import QColor

class ColoredTableWidget(TableWidget):
    """Customized table widget based on the napari_skimage_regionprops TableWidget
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ascending = False # for choosing whether to sort ascending or descending

        # Reconnect the clicked signal to your custom method.
        self._view.clicked.connect(self._clicked_table)

        # Connect to single click in the header to sort the table.
        self._view.horizontalHeader().sectionClicked.connect(self._sort_table)


    def _set_label_colors_to_rows(self) -> None:
        """Apply the colors of the napari label image to the table"""

        for i in range(self._view.rowCount()):
            label = self._table['label'][i]
            label_color = to_rgb(self._layer.get_color(label))
            scaled_color = (int(label_color[0] * 255), int(label_color[1] * 255), int(label_color[2] * 255))
            for j in range(self._view.columnCount()):
                self._view.item(i, j).setBackground(QColor(*scaled_color))
    
    def _clicked_table(self):
        """Also set show_selected_label to True and jump to the corresponding stack position"""
        
        super()._clicked_table()
        self._layer.show_selected_label = True

        row = self._view.currentRow()
        z = int(self._table["centroid-0"][row])
        current_step = self._viewer.dims.current_step
        if len(current_step) == 4:
            new_step = (current_step[0], z, current_step[2], current_step[3])
        elif len(current_step) == 3: 
            new_step = (z, current_step[1], current_step[2])
        else: 
            new_step = current_step
        self._viewer.dims.current_step = new_step
    
    def _sort_table(self):
        """Sorts the table in ascending or descending order"""

        selected_column = list(self._table.keys())[self._view.currentColumn()]
        df = pd.DataFrame(self._table).sort_values(by=selected_column, ascending=self.ascending)
        self.ascending = not self.ascending

        self.set_content(df.to_dict(orient='list'))
        self._set_label_colors_to_rows()


