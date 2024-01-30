"""
Napari plugin widget for editing N-dimensional label data
"""

import os
import shutil
import functools
import napari
import tifffile
import numpy        as np
import dask.array   as da

from scipy.ndimage                          import binary_erosion
from scipy                                  import ndimage
from dask_image.imread                      import imread
from typing                                 import Tuple

from napari.layers                          import Image, Labels, Points
from skimage                                import measure
from skimage.io                             import imread
from skimage.segmentation                   import expand_labels

from qtpy.QtWidgets                         import QScrollArea, QGroupBox, QMessageBox, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLineEdit, QSpinBox, QComboBox, QTabWidget
from qtpy.QtCore                            import *
from PyQt5.QtCore                           import pyqtSignal

from ._custom_table_widget                  import ColoredTableWidget
from .napari_multiple_view_widget           import CrossWidget, MultipleViewerWidget

class LayerDropdown(QComboBox):
    """QComboBox widget with functions for updating the selected layer and to update the list of options when the list of layers is modified. 
    
    """
    
    layer_changed = pyqtSignal(str)  # Define a signal to emit the selected layer name

    def __init__(self, viewer, layer_type:Tuple):
        super().__init__()
        self.viewer = viewer
        self.layer_type = layer_type
        self.viewer.layers.events.changed.connect(self._update_dropdown)
        self.viewer.layers.events.inserted.connect(self._update_dropdown)
        self.viewer.layers.events.removed.connect(self._update_dropdown)
        self.viewer.layers.selection.events.changed.connect(self._on_selection_changed)
        self.currentIndexChanged.connect(self._emit_layer_changed)
        self._update_dropdown()
    
    def _on_selection_changed(self) -> None:
        """Request signal emission if the user changes the layer selection."""           

        if len(self.viewer.layers.selection) == 1:  # Only consider single layer selection
            selected_layer = self.viewer.layers.selection.active
            if isinstance(selected_layer, self.layer_type):
                self.setCurrentText(selected_layer.name)
                self._emit_layer_changed()        

    def _update_dropdown(self) -> None:
        """Update the list of options in the dropdown menu whenever the list of layers is changed"""

        selected_layer = self.currentText()
        self.clear()
        layers = [layer for layer in self.viewer.layers if isinstance(layer, self.layer_type) and not layer.name == "label options"]
        items = []
        for layer in layers:
            self.addItem(layer.name)
            items.append(layer.name)
        
        # In case the currently selected layer is one of the available items, set it again to the current value of the dropdown. 
        if selected_layer in items:
            self.setCurrentText(selected_layer)

    def _emit_layer_changed(self):
        """Emit a signal holding the currently selected layer"""

        selected_layer = self.currentText()
        self.layer_changed.emit(selected_layer)

class AnnotateLabelsND(QWidget):
    """Widget for manual correction of label data, for example to prepare ground truth data for training a segmentation model
    
    """
    
    def __init__(self, viewer: 'napari.viewer.Viewer') -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear() # ensure viewer is clean

        self.labels = None
        self.source_labels = None
        self.target_labels = None
        self.table = None
        self.points = None
        self.copy_points = None
        self.outputdir = None
        self.settings_layout = QVBoxLayout()
        self.tab_widget = QTabWidget(self)
        self.option_labels = None

        ### specify output directory
        outputbox_layout = QHBoxLayout()
        self.outputdirbtn = QPushButton('Select output directory')
        self.output_path = QLineEdit()
        outputbox_layout.addWidget(self.outputdirbtn)
        outputbox_layout.addWidget(self.output_path)
        self.outputdirbtn.clicked.connect(self._on_get_output_dir)
        self.settings_layout.addLayout(outputbox_layout)

        ### create the dropdown for selecting label images
        self.label_dropdown = LayerDropdown(self.viewer, (Labels))
        self.label_dropdown.layer_changed.connect(self._update_labels)
        self.settings_layout.addWidget(self.label_dropdown)

        ### Add option to convert dask array to in-memory array
        self.convert_to_array_btn = QPushButton('Convert to in-memory array')
        self.convert_to_array_btn.setEnabled(self.labels != None and type(self.labels.data) == da.core.Array)
        self.convert_to_array_btn.clicked.connect(self._convert_to_array)
        self.settings_layout.addWidget(self.convert_to_array_btn)

        ### Add widget for adding overview table
        self.table_btn = QPushButton('Show table')
        self.table_btn.clicked.connect(self._create_summary_table)
        self.table_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0)) 
        if self.labels is not None:
            self.table_btn.setEnabled(True)
        self.settings_layout.addWidget(self.table_btn)

        ## Add save labels widget
        self.save_btn = QPushButton('Save labels')
        self.save_btn.clicked.connect(self._save_labels)
        self.settings_layout.addWidget(self.save_btn)

        ## Add button to clear all layers
        self.clear_btn = QPushButton('Clear all layers')
        self.clear_btn.clicked.connect(self._clear_layers)
        self.settings_layout.addWidget(self.clear_btn)

        ### Add widget for filtering by points layer
        point_filter_box = QGroupBox('Select objects with points')
        point_filter_layout = QVBoxLayout()
        self.point_dropdown = LayerDropdown(self.viewer, (Points))
        self.point_dropdown.layer_changed.connect(self._update_points)

        remove_keep_btn_layout = QHBoxLayout()
        self.keep_pts_btn = QPushButton('Keep')
        self.keep_pts_btn.clicked.connect(self._keep_objects)
        self.remove_pts_btn = QPushButton('Remove')
        self.remove_pts_btn.clicked.connect(self._delete_objects)
        remove_keep_btn_layout.addWidget(self.keep_pts_btn)
        remove_keep_btn_layout.addWidget(self.remove_pts_btn)

        point_filter_layout.addWidget(self.point_dropdown)
        point_filter_layout.addLayout(remove_keep_btn_layout)

        point_filter_box.setLayout(point_filter_layout)
        self.settings_layout.addWidget(point_filter_box)

        ### Add widget for copy-pasting labels from one layer to another
        copy_labels_box = QGroupBox('Copy-paste labels')
        copy_labels_layout = QVBoxLayout()

        add_option_layer_btn = QPushButton('Add layer with different label options from folder')
        add_option_layer_btn.clicked.connect(self._add_option_layer)

        copy_labels_layout.addWidget(add_option_layer_btn)

        copy_labels_box.setLayout(copy_labels_layout)
        self.settings_layout.addWidget(copy_labels_box)

        ### Add widget for size filtering
        filterbox = QGroupBox('Filter objects by size')
        filter_layout = QVBoxLayout()

        label_size = QLabel("Size threshold (voxels)")
        threshold_size_layout = QHBoxLayout()
        self.min_size_field = QSpinBox()
        self.min_size_field.setMaximum(1000000)
        self.delete_btn = QPushButton('Delete')
        threshold_size_layout.addWidget(self.min_size_field)
        threshold_size_layout.addWidget(self.delete_btn)

        filter_layout.addWidget(label_size)
        filter_layout.addLayout(threshold_size_layout)
        self.delete_btn.clicked.connect(self._delete_small_objects)
        self.delete_btn.setEnabled(True)

        filterbox.setLayout(filter_layout)
        self.settings_layout.addWidget(filterbox)

        self.setLayout(self.settings_layout)

        ### Add widget for smoothing labels
        smoothbox = QGroupBox('Smooth objects')
        smooth_boxlayout = QVBoxLayout()

        smooth_layout = QHBoxLayout()
        self.median_radius_field = QSpinBox()
        self.median_radius_field.setMaximum(100)
        self.smooth_btn = QPushButton('Smooth')
        smooth_layout.addWidget(self.median_radius_field)
        smooth_layout.addWidget(self.smooth_btn)

        smooth_boxlayout.addWidget(QLabel("Median filter radius"))
        smooth_boxlayout.addLayout(smooth_layout)

        self.smooth_btn.clicked.connect(self._smooth_objects)
        self.smooth_btn.setEnabled(True)

        smoothbox.setLayout(smooth_boxlayout)
        self.settings_layout.addWidget(smoothbox)

        ### Add widget for eroding/dilating labels
        dil_erode_box = QGroupBox('Erode/dilate labels')
        dil_erode_box_layout = QVBoxLayout()

        radius_layout = QHBoxLayout()
        str_element_diameter_label = QLabel("Structuring element diameter")
        str_element_diameter_label.setFixedWidth(200)
        self.structuring_element_diameter = QSpinBox()
        self.structuring_element_diameter.setMaximum(100)
        self.structuring_element_diameter.setValue(1)
        radius_layout.addWidget(str_element_diameter_label)
        radius_layout.addWidget(self.structuring_element_diameter)

        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Iterations")
        iterations_label.setFixedWidth(200)
        self.iterations = QSpinBox()
        self.iterations.setMaximum(100)
        self.iterations.setValue(1)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations)

        shrink_dilate_buttons_layout = QHBoxLayout()
        self.erode_btn = QPushButton('Erode')
        self.dilate_btn = QPushButton('Dilate')
        self.erode_btn.clicked.connect(self._erode_labels)
        self.dilate_btn.clicked.connect(self._dilate_labels)
        shrink_dilate_buttons_layout.addWidget(self.erode_btn)
        shrink_dilate_buttons_layout.addWidget(self.dilate_btn)

        if self.labels is not None:
            self.erode_btn.setEnabled(True)
            self.dilate_btn.setEnabled(True)

        dil_erode_box_layout.addLayout(radius_layout)
        dil_erode_box_layout.addLayout(iterations_layout)
        dil_erode_box_layout.addLayout(shrink_dilate_buttons_layout)

        dil_erode_box.setLayout(dil_erode_box_layout)
        self.settings_layout.addWidget(dil_erode_box)

        ### Threshold image
        threshold_box = QGroupBox('Threshold')
        threshold_box_layout = QVBoxLayout()

        self.threshold_layer_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.threshold_layer_dropdown.layer_changed.connect(self._update_threshold_layer)
        threshold_box_layout.addWidget(self.threshold_layer_dropdown)

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(QLabel('Min value'))
        self.min_threshold = QSpinBox()
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(QLabel('Max value'))
        self.max_threshold = QSpinBox()
        self.max_threshold.setMaximum(65535)
        self.max_threshold.setValue(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        threshold_box_layout.addLayout(min_threshold_layout)
        threshold_box_layout.addLayout(max_threshold_layout)
        threshold_btn = QPushButton('Run')
        threshold_btn.clicked.connect(self._threshold)
        threshold_box_layout.addWidget(threshold_btn)
        
        threshold_box.setLayout(threshold_box_layout)
        self.settings_layout.addWidget(threshold_box)

        ### Add one image to another
        image_calc_box = QGroupBox('Image calculator')
        image_calc_box_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_layout.addWidget(QLabel('Label image 1'))
        self.image1_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image1_dropdown.layer_changed.connect(self._update_image1)
        image1_layout.addWidget(self.image1_dropdown)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel('Label image 2'))
        self.image2_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image2_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.image2_dropdown)

        image_calc_box_layout.addLayout(image1_layout)
        image_calc_box_layout.addLayout(image2_layout)

        operation_layout = QHBoxLayout()
        self.operation = QComboBox()
        self.operation.addItem("Add")
        self.operation.addItem("Subtract")
        self.operation.addItem("Multiply")
        self.operation.addItem("Divide")
        self.operation.addItem("AND")
        self.operation.addItem("OR")
        operation_layout.addWidget(QLabel('Operation'))
        operation_layout.addWidget(self.operation)
        image_calc_box_layout.addLayout(operation_layout)

        add_images_btn = QPushButton('Run')
        add_images_btn.clicked.connect(self._calculate_images)
        image_calc_box_layout.addWidget(add_images_btn)

        image_calc_box.setLayout(image_calc_box_layout)
        self.settings_layout.addWidget(image_calc_box)

        ### add the button to show the cross in multiview
        cross_box = QGroupBox('Add cross to multiview')
        cross_box_layout = QHBoxLayout()
        self.cross = CrossWidget(self.viewer)
        self.cross.setChecked(False)
        self.cross.layer = None
        cross_box_layout.addWidget(self.cross)
        cross_box.setLayout(cross_box_layout)
        self.settings_layout.addWidget(cross_box)

        ### combine into tab widget
    
        ## Add multiview widget
        self.multi_view_table_widget = QWidget()
        self.multi_view_table_layout = QHBoxLayout()
        self.multiview_widget = MultipleViewerWidget(self.viewer)
        self.multi_view_table_layout.addWidget(self.multiview_widget)
        self.multi_view_table_widget.setLayout(self.multi_view_table_layout)
        self.tab_widget.addTab(self.multi_view_table_widget, "Orthogonal Views")

        ## add combined settings widgets
        self.settings_widgets = QWidget()
        self.settings_widgets.setLayout(self.settings_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.settings_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Settings")

        # Add the tab widget to the main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.setLayout(self.main_layout)

    def _on_get_output_dir(self) -> None:
        """Show a dialog window to let the user pick the output directory."""
        
        path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if path:
            self.output_path.setText(path)
            self.outputdir = str(self.output_path.text())
        
    def _update_labels(self, selected_layer) -> None:
        """Update the layer that is set to be the 'labels' layer that is being edited."""

        if selected_layer == '':
            self.labels = None
        else:
            self.labels = self.viewer.layers[selected_layer]
            self.label_dropdown.setCurrentText(selected_layer)
            self.convert_to_array_btn.setEnabled(type(self.labels.data) == da.core.Array)
    
    def _update_source_labels(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == '':
            self.source_labels = None
        else:
            self.source_labels = self.viewer.layers[selected_layer]
            self.source_label_dropdown.setCurrentText(selected_layer)
    
    def _update_target_labels(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'target labels' layer for copying labels to."""

        if selected_layer == '':
            self.target_labels = None
        else:
            self.target_labels = self.viewer.layers[selected_layer]
            self.target_label_dropdown.setCurrentText(selected_layer)

    def _update_points(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'points' layer for picking labels."""

        if selected_layer == '':
            self.points = None
        else:
            self.points = self.viewer.layers[selected_layer]
            self.point_dropdown.setCurrentText(selected_layer)
    
    def _update_copy_points(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'points' layer for copying labels from one layer to another."""

        if selected_layer == '':
            self.copy_points = None
        else:
            self.copy_points = self.viewer.layers[selected_layer]
            self.copy_point_dropdown.setCurrentText(selected_layer)
    
    def _update_threshold_layer(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == '':
            self.threshold_layer = None
        else:
            self.threshold_layer = self.viewer.layers[selected_layer]
            self.threshold_layer_dropdown.setCurrentText(selected_layer)

    def _update_image1(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == '':
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

    def _update_image2(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == '':
            self.image2_layer = None
        else:
            self.image2_layer = self.viewer.layers[selected_layer]
            self.image2_dropdown.setCurrentText(selected_layer)

    def _convert_to_array(self) -> None: 
        """Convert from dask array to in-memory array. This is necessary for manual editing using the label tools (brush, eraser, fill bucket)."""
        
        if type(self.labels.data) == da.core.Array:
            stack = []
            for i in range(self.labels.data.shape[0]):
                current_stack = self.labels.data[i].compute()
                stack.append(current_stack)
            self.labels.data = np.stack(stack, axis = 0)
    
    def _create_summary_table(self) -> None:
        """Create table displaying the sizes of the different labels in the current stack"""

        if type(self.labels.data) == da.core.Array:
            tp = self.viewer.dims.current_step[0]
            current_stack = self.labels.data[tp].compute()  # Compute the current stack
            props = measure.regionprops_table(current_stack, properties = ['label', 'area', 'centroid'])
            if hasattr(self.labels, "properties"):
                self.labels.properties = props
            if hasattr(self.labels, "features"):
                self.labels.features = props

        else:
            if len(self.labels.data.shape) == 4:
                tp = self.viewer.dims.current_step[0]
                props = measure.regionprops_table(self.labels.data[tp], properties = ['label', 'area', 'centroid'])
                if hasattr(self.labels, "properties"):
                    self.labels.properties = props
                if hasattr(self.labels, "features"):
                    self.labels.features = props
            
            elif len(self.labels.data.shape) == 3: 
                props = measure.regionprops_table(self.labels.data, properties = ['label', 'area', 'centroid'])
                if hasattr(self.labels, "properties"):
                    self.labels.properties = props
                if hasattr(self.labels, "features"):
                    self.labels.features = props
            else: 
                print('input should be a 3D or 4D array')
                self.table = None

        # add the table to the viewer, using the code from napari-skimage-regionprops
        if self.table is not None:
            self.table.hide()

        if self.viewer is not None:
            self.table = ColoredTableWidget(self.labels, self.viewer)
            self.table._set_label_colors_to_rows()
            self.table.setMinimumWidth(500)
            self.multi_view_table_layout.addWidget(self.table)
 
    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        if type(self.labels.data) == da.core.Array:

            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False
            
            else:
                outputdir = os.path.join(self.outputdir, (self.labels.name + "_finalresult"))
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(self.labels.data.shape[0]):  # Loop over the first dimension
                        current_stack = self.labels.data[i].compute()  # Compute the current stack                 
                        tifffile.imwrite(os.path.join(outputdir, (self.labels.name + '_TP' + str(i).zfill(4) + '.tif')), np.array(current_stack, dtype = 'uint16'))
                return True

        elif len(self.labels.data.shape) == 4:
            filename, _ = QFileDialog.getSaveFileName(
            caption='Save Labels',
            directory='',
            filter='TIFF files (*.tif *.tiff)')
            for i in range(self.labels.data.shape[0]):
                labels_data = self.labels.data[i].astype(np.uint16)
                tifffile.imwrite((filename.split('.tif')[0] + '_TP' + str(i).zfill(4) + '.tif'), labels_data)

        elif len(self.labels.data.shape) == 3: 
            filename, _ = QFileDialog.getSaveFileName(
                caption='Save Labels',
                directory='',
                filter='TIFF files (*.tif *.tiff)')

            if filename:
                labels_data = self.labels.data.astype(np.uint16)
                tifffile.imwrite(filename, labels_data)
        
        else: 
            print('labels should be a 3D or 4D array')

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        if self.table is not None:
            self.table.hide()
            self.table = None
            self.settings_layout.update()

        self.cross.setChecked(False)
        self.cross.layer = None
        self.viewer.layers.clear()

    def _keep_objects(self) -> None:
        """Keep only the labels that are selected by the points layer."""

        if type(self.labels.data) == da.core.Array:
            tps = np.unique([int(p[0]) for p in self.points.data])
            for tp in tps:
                labels_to_keep = []               
                points = [p for p in self.points.data if p[0] == tp]
                current_stack = self.labels.data[tp].compute()  # Compute the current stack
                for p in points:
                    labels_to_keep.append(current_stack[int(p[1]), int(p[2]), int(p[3])])
                mask = functools.reduce(np.logical_or, (current_stack==val for val in labels_to_keep))
                filtered = np.where(mask, current_stack, 0)
                self.labels.data[tp] = filtered
            self.labels.data = self.labels.data # to trigger viewer update

        else:                 
            if len(self.points.data[0]) == 4:
                tps = np.unique([int(p[0]) for p in self.points.data])
                for tp in tps:
                    labels_to_keep = []
                    points = [p for p in self.points.data if p[0] == tp]
                    for p in points:
                        labels_to_keep.append(self.labels.data[tp, int(p[1]), int(p[2]), int(p[3])])
                    mask = functools.reduce(np.logical_or, (self.labels.data[tp]==val for val in labels_to_keep))
                    filtered = np.where(mask, self.labels.data[tp], 0)
                    self.labels.data[tp] = filtered
                self.labels.data = self.labels.data # to trigger viewer update

            else:            
                labels_to_keep = []
                for p in self.points.data:
                    if len(p) == 2:
                        labels_to_keep.append(self.labels.data[int(p[0]), int(p[1])])
                    elif len(p) == 3:
                        labels_to_keep.append(self.labels.data[int(p[0]), int(p[1]), int(p[2])])

                mask = functools.reduce(np.logical_or, (self.labels.data==val for val in labels_to_keep))
                filtered = np.where(mask, self.labels.data, 0)
                
                self.labels = self.viewer.add_labels(filtered, name = self.labels.name + '_points_kept')
                self._update_labels(self.labels.name)

    def _delete_objects(self) -> None:
        """Delete all labels selected by the points layer."""

        if type(self.labels.data) == da.core.Array:
            tps = np.unique([int(p[0]) for p in self.points.data])
            for tp in tps:
                labels_to_keep = []               
                points = [p for p in self.points.data if p[0] == tp]
                current_stack = self.labels.data[tp].compute()  # Compute the current stack
                for p in points:
                    labels_to_keep.append(current_stack[int(p[1]), int(p[2]), int(p[3])])
                mask = functools.reduce(np.logical_or, (current_stack==val for val in labels_to_keep))
                inverse_mask = np.logical_not(mask)
                filtered = np.where(inverse_mask, current_stack, 0)
                self.labels.data[tp] = filtered
            self.labels.data = self.labels.data

        else:
            if len(self.points.data[0]) == 4:
                tps = np.unique([int(p[0]) for p in self.points.data])
                for tp in tps:
                    labels_to_keep = []
                    points = [p for p in self.points.data if p[0] == tp]
                    for p in points:
                        labels_to_keep.append(self.labels.data[tp, int(p[1]), int(p[2]), int(p[3])])
                    mask = functools.reduce(np.logical_or, (self.labels.data[tp]==val for val in labels_to_keep))
                    inverse_mask = np.logical_not(mask)
                    filtered = np.where(inverse_mask, self.labels.data[tp], 0)
                    self.labels.data[tp] = filtered
                self.labels.data = self.labels.data # to trigger viewer update
            
            else: 
                labels_to_keep = []
                for p in self.points.data:
                    if len(p) == 2:
                        labels_to_keep.append(self.labels.data[int(p[0]), int(p[1])])
                    elif len(p) == 3:
                        labels_to_keep.append(self.labels.data[int(p[0]), int(p[1]), int(p[2])])
                    elif len(p) == 4:
                        labels_to_keep.append(self.labels.data[int(p[0]), int(p[1]), int(p[2], int(p[3]))])

                mask = functools.reduce(np.logical_or, (self.labels.data==val for val in labels_to_keep))
                inverse_mask = np.logical_not(mask)
                filtered = np.where(inverse_mask, self.labels.data, 0)

                self.labels = self.viewer.add_labels(filtered, name = self.labels.name + '_points_removed')
                self._update_labels(self.labels.name)


    def _add_option_layer(self):
        """Add a new labels layer that contains different alternative segmentations as channels, and add a function to select and copy these cells through shift-clicking"""

        path = QFileDialog.getExistingDirectory(self, 'Select Label Image Parent Folder')
        if path:
            label_dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            label_stacks = []
            for d in label_dirs:
                # n dirs indicates number of channels
                label_files = [f for f in os.listdir(os.path.join(path, d)) if '.tif' in f]
                label_imgs = []
                for f in label_files:
                    # n label_files indicates n time points
                    img = imread(os.path.join(path, d, f))
                    label_imgs.append(img)
                
                if len(label_imgs) > 1:
                    label_stack = np.stack(label_imgs, axis = 0)
                    label_stacks.append(label_stack)               
                else:
                    label_stacks.append(img)
            
            if len(label_stacks) > 1:
                self.option_labels = np.stack(label_stacks, axis = 0)
            elif len(label_stacks) == 1:
                self.option_labels = label_stacks[0]
            
            n_channels = len(label_dirs)
            n_timepoints = len(label_files)
            if len(img.shape) == 3: 
                n_slices = img.shape[0]
            elif len(img.shape) == 2:
                n_slices = 1
            
            self.option_labels = self.option_labels.reshape(n_channels, n_timepoints, n_slices, img.shape[-2], img.shape[-1])    
            self.option_labels = self.viewer.add_labels(self.option_labels, name = 'label options')

        viewer = self.viewer
        @viewer.mouse_drag_callbacks.append
        def cell_copied(viewer, event):
            if event.type == "mouse_press" and 'Shift' in event.modifiers and viewer.layers.selection.active == self.option_labels:
                coords = self.option_labels.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = self.option_labels.get_value(coords)
                mask = self.option_labels.data[coords[0], coords[1], :, :, :] == selected_label

                if type(self.labels.data) == da.core.Array:
                    target_stack = self.labels.data[coords[-4]].compute()
                    orig_label = target_stack[coords[-3], coords[-2], coords[-1]]
                    if orig_label != 0: 
                        target_stack[target_stack == orig_label] = 0  
                    target_stack[mask] = np.max(target_stack) + 1
                    self.labels.data[coords[-4]] = target_stack
                    self.labels.data = self.labels.data
                
                else: 
                    if len(self.labels.data.shape) == 3:
                        orig_label = self.labels.data[coords[-3], coords[-2], coords[-1]]

                        if orig_label != 0:
                            self.labels.data[self.labels.data == orig_label] = 0 # set the original label to zero                
                        self.labels.data[mask] = np.max(self.labels.data) + 1
                        self.labels.data = self.labels.data

                    elif len(self.labels.data.shape) == 4:
                        orig_label = self.labels.data[coords[-4], coords[-3], coords[-2], coords[-1]]

                        if orig_label != 0: 
                            self.labels.data[coords[-4]][self.labels.data[coords[-4]] == orig_label] = 0 # set the original label to zero                  
                        self.labels.data[coords[-4]][mask] = np.max(self.labels.data) + 1
                        self.labels.data = self.labels.data
                    
                    elif len(self.labels.data.shape) == 5:
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setText("Copy-pasting in 5 dimensions is not implemented, do you want to convert the labels layer to 5 dimensions (tzyx)?")
                        msg_box.setWindowTitle("Convert to 4 dimensions?")

                        yes_button = msg_box.addButton(QMessageBox.Yes)
                        no_button = msg_box.addButton(QMessageBox.No)

                        msg_box.exec_()

                        if msg_box.clickedButton() == yes_button:
                            self.labels.data = self.labels.data[0]
                        elif msg_box.clickedButton() == no_button:
                            return False                  
                    else:
                        print('copy-pasting in more than 5 dimensions is not supported')
    
    def _delete_small_objects(self) -> None:
        """Delete small objects in the selected layer"""

        if type(self.labels.data) == da.core.Array:
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

            else:
                outputdir = os.path.join(self.outputdir, (self.labels.name + "_sizefiltered"))
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(self.labels.data.shape[0]):  # Loop over the first dimension
                    current_stack = self.labels.data[i].compute()  # Compute the current stack

                    # measure the sizes in pixels of the labels in slice using skimage.regionprops
                    props = measure.regionprops(current_stack)
                    filtered_labels = [p.label for p in props if p.area > self.min_size_field.value()]
                    mask = functools.reduce(np.logical_or, (current_stack==val for val in filtered_labels))
                    filtered = np.where(mask, current_stack, 0)
                    tifffile.imwrite(os.path.join(outputdir, (self.labels.name + '_sizefiltered_TP' + str(i).zfill(4) + '.tif')), np.array(filtered, dtype = 'uint16'))
                
                file_list = [os.path.join(outputdir, fname) for fname in os.listdir(outputdir) if fname.endswith('.tif')]
                self.labels = self.viewer.add_labels(da.stack([imread(fname) for fname in sorted(file_list)]), name = self.labels.name + '_sizefiltered')
                self._update_labels(self.labels.name)
       
        else:
            # Image data is a normal array and can be directly edited. 
            if len(self.labels.data.shape) == 4: 
                stack = []
                for i in range(self.labels.data.shape[0]):
                    props = measure.regionprops(self.labels.data[i])
                    filtered_labels = [p.label for p in props if p.area > self.min_size_field.value()]
                    mask = functools.reduce(np.logical_or, (self.labels.data[i]==val for val in filtered_labels))
                    filtered = np.where(mask, self.labels.data[i], 0)
                    stack.append(filtered)
                self.labels = self.viewer.add_labels(np.stack(stack, axis = 0), name = self.labels.name + '_sizefiltered')
                self._update_labels(self.labels.name)
            
            elif len(self.labels.data.shape) == 3:
                props = measure.regionprops(self.labels.data)
                filtered_labels = [p.label for p in props if p.area > self.min_size_field.value()]
                mask = functools.reduce(np.logical_or, (self.labels.data==val for val in filtered_labels))
                self.labels = self.viewer.add_labels(np.where(mask, self.labels.data, 0), name = self.labels.name + '_sizefiltered')
                self._update_labels(self.labels.name)

            else:
                print('input should be 3D or 4D array')

    def _smooth_objects(self) -> None:
        """Smooth objects by using a median filter."""
        
        if type(self.labels.data) == da.core.Array:
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False
            
            else:
                outputdir = os.path.join(self.outputdir, (self.labels.name + "_smoothed"))
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)
                
                for i in range(self.labels.data.shape[0]):  # Loop over the first dimension
                    current_stack = self.labels.data[i].compute()  # Compute the current stack
                    smoothed = ndimage.median_filter(current_stack, size = self.median_radius_field.value())
                    tifffile.imwrite(os.path.join(outputdir, (self.labels.name + '_smoothed_TP' + str(i).zfill(4) + '.tif')), np.array(smoothed, dtype = 'uint16'))
                
                file_list = [os.path.join(outputdir, fname) for fname in os.listdir(outputdir) if fname.endswith('.tif')]
                self.labels = self.viewer.add_labels(da.stack([imread(fname) for fname in sorted(file_list)]), name = self.labels.name + '_smoothed')
                self._update_labels(self.labels.name)

        else:
            if len(self.labels.data.shape) == 4: 
                stack = []
                for i in range(self.labels.data.shape[0]):
                    smoothed = ndimage.median_filter(self.labels.data[i], size = self.median_radius_field.value())
                    stack.append(smoothed)
                self.labels = self.viewer.add_labels(np.stack(stack, axis = 0), name = self.labels.name + '_smoothed')
                self._update_labels(self.labels.name)  

            elif len(self.labels.data.shape) == 3:
                self.labels = self.viewer.add_labels(ndimage.median_filter(self.labels.data, size = self.median_radius_field.value()), name = self.labels.name + '_smoothed')
                self._update_labels(self.labels.name)
            
            else: 
                print('input should be a 3D or 4D array')

    def _erode_labels(self):
        """Shrink oversized labels through erosion"""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()
        structuring_element = np.ones((diam, diam, diam), dtype=bool)  # Define a 3x3x3 structuring element for 3D erosion

        if type(self.labels.data) == da.core.Array:
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False
            
            else:
                outputdir = os.path.join(self.outputdir, (self.labels.name + "_eroded"))
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(self.labels.data.shape[0]):  # Loop over the first dimension
                    current_stack = self.labels.data[i].compute()  # Compute the current stack
                    mask = current_stack > 0
                    filled_mask = ndimage.binary_fill_holes(mask)
                    eroded_mask = binary_erosion(filled_mask, structure=structuring_element, iterations=iterations)
                    eroded = np.where(eroded_mask, current_stack, 0)
                    tifffile.imwrite(os.path.join(outputdir, (self.labels.name + '_eroded_TP' + str(i).zfill(4) + '.tif')), np.array(eroded, dtype = 'uint16'))
                
                file_list = [os.path.join(outputdir, fname) for fname in os.listdir(outputdir) if fname.endswith('.tif')]
                self.labels = self.viewer.add_labels(da.stack([imread(fname) for fname in sorted(file_list)]), name = self.labels.name + '_eroded')               
                self._update_labels(self.labels.name)
                return True

        else:
            if len(self.labels.data.shape) == 4: 
                stack = []
                for i in range(self.labels.data.shape[0]):
                    mask = self.labels.data[i] > 0
                    filled_mask = ndimage.binary_fill_holes(mask)
                    eroded_mask = binary_erosion(filled_mask, structure=structuring_element, iterations=iterations)
                    stack.append(np.where(eroded_mask, self.labels.data[i], 0))
                self.labels = self.viewer.add_labels(np.stack(stack, axis = 0), name = self.labels.name + '_eroded')
                self._update_labels(self.labels.name)    
            elif len(self.labels.data.shape) == 3:     
                mask = self.labels.data > 0
                filled_mask = ndimage.binary_fill_holes(mask)
                eroded_mask = binary_erosion(filled_mask, structure=structuring_element, iterations=iterations)
                self.labels = self.viewer.add_labels(np.where(eroded_mask, self.labels.data, 0), name = self.labels.name + '_eroded')
                self._update_labels(self.labels.name)
            else: 
                print('4D or 3D array required!')

    def _dilate_labels(self):
        """Dilate labels in the selected layer."""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()

        if type(self.labels.data) == da.core.Array:
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False
            
            else:
                outputdir = os.path.join(self.outputdir, (self.labels.name + "_dilated"))
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(self.labels.data.shape[0]):  # Loop over the first dimension
                    expanded_labels = self.labels.data[i].compute()  # Compute the current stack
                    for j in range(iterations):
                        expanded_labels = expand_labels(expanded_labels, distance = diam)
                    tifffile.imwrite(os.path.join(outputdir, (self.labels.name + '_dilated_TP' + str(i).zfill(4) + '.tif')), np.array(expanded_labels, dtype = 'uint16'))
                
                file_list = [os.path.join(outputdir, fname) for fname in os.listdir(outputdir) if fname.endswith('.tif')]
                self.labels = self.viewer.add_labels(da.stack([imread(fname) for fname in sorted(file_list)]), name = self.labels.name + '_dilated')
                self._update_labels(self.labels.name)
                return True

        else: 
            if len(self.labels.data.shape) == 4: 
                stack = []
                for i in range(self.labels.data.shape[0]):
                    expanded_labels = self.labels.data[i]
                    for i in range(iterations):
                        expanded_labels = expand_labels(expanded_labels, distance = diam)                  
                    stack.append(expanded_labels)
                self.labels = self.viewer.add_labels(np.stack(stack, axis = 0), name = self.labels.name + '_dilated')
                self._update_labels(self.labels.name)  

            elif len(self.labels.data.shape) == 3:
                expanded_labels = self.labels.data
                for i in range(iterations):
                    expanded_labels = expand_labels(expanded_labels, distance = diam)

                self.labels = self.viewer.add_labels(expanded_labels, name = self.labels.name + '_dilated')
                self._update_labels(self.labels.name)
            else: 
                print('input should be a 3D or 4D stack')
    
    def _threshold(self): 
        """Threshold the selected label or intensity image"""

        if type(self.threshold_layer.data) == da.core.Array:
                msg = QMessageBox()
                msg.setWindowTitle("Thresholding not yet implemented for dask arrays")
                msg.setText("Thresholding not yet implemented for dask arrays")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

        thresholded = (self.threshold_layer.data >= int(self.min_threshold.value())) & (self.threshold_layer.data <= int(self.max_threshold.value()))
        self.viewer.add_labels(thresholded, name = self.threshold_layer.name + "_thresholded")

    def _calculate_images(self):
        """Add label image 2 to label image 1"""

        if type(self.image1_layer) == da.core.Array or type(self.image2_layer) == da.core.Array:
                msg = QMessageBox()
                msg.setWindowTitle("Cannot yet run image calculator on dask arrays")
                msg.setText("Cannot yet run image calculator on dask arrays")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False
        if self.image1_layer.data.shape != self.image2_layer.data.shape: 
                msg = QMessageBox()
                msg.setWindowTitle("Images must have the same shape")
                msg.setText("Images must have the same shape")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False
              
        if self.operation.currentText() == "Add":
            self.viewer.add_image(np.add(self.image1_layer.data, self.image2_layer.data))
        if self.operation.currentText() == "Subtract":
            self.viewer.add_image(np.subtract(self.image1_layer.data, self.image2_layer.data))
        if self.operation.currentText() == "Multiply":
            self.viewer.add_image(np.multiply(self.image1_layer.data, self.image2_layer.data))
        if self.operation.currentText() == "Divide":
            self.viewer.add_image(np.divide(self.image1_layer.data, self.image2_layer.data, out=np.zeros_like(self.image1_layer.data, dtype=float), where=self.image2_layer.data!=0))
        if self.operation.currentText() == "AND":
            self.viewer.add_labels(np.logical_and(self.image1_layer.data != 0, self.image2_layer.data != 0).astype(int))
        if self.operation.currentText() == "OR":
            self.viewer.add_labels(np.logical_or(self.image1_layer.data != 0, self.image2_layer.data != 0).astype(int))

