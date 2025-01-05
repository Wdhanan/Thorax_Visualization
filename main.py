# Standardbibliotheken
import sys

# Drittanbieter-Bibliotheken
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
    QComboBox, QSlider, QLabel, QDialog, QMessageBox, QRadioButton
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# VTK-Bibliotheken
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.numpy_interface import dataset_adapter as dsa  # type: ignore

# Funktion zum Laden der .vti Datei
def load_vti_file(filepath):
    """Lädt die .vti Datei mit den medizinischen Bilddaten."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()

# Funktion, um die Voxel-Daten in ein NumPy-Array zu konvertieren
def extract_voxel_data(image_data):
    """Extrahiert die Voxel-Daten als NumPy-Array."""
    point_data = image_data.GetPointData().GetScalars()
    np_array = dsa.WrapDataObject(image_data).PointData[point_data.GetName()]
    return np_array

class HistogramDialog(QDialog):
    def __init__(self, all_histogram_values, roi_histogram_values=None):
        super().__init__()
        self.setWindowTitle("Histogramm")
        self.setGeometry(100, 100, 800, 600)

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)  # Create the canvas here

        # Radiobuttons für Hintergrundfarbe
        self.bg_white_radio = QRadioButton("Weiß")
        self.bg_black_radio = QRadioButton("Schwarz")
        self.bg_white_radio.setChecked(True)  # Standardmäßig Weiß
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Hintergrund:"))
        bg_layout.addWidget(self.bg_white_radio)
        bg_layout.addWidget(self.bg_black_radio)

        # Histogramm zeichnen (mit initialem weißen Hintergrund)
        self.update_histogram(all_histogram_values, roi_histogram_values)

        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addLayout(bg_layout)  # Radiobuttons zum Layout hinzufügen
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Verbinde Radiobuttons mit Update-Funktion
        self.bg_white_radio.toggled.connect(lambda: self.update_histogram(all_histogram_values, roi_histogram_values))
        self.bg_black_radio.toggled.connect(lambda: self.update_histogram(all_histogram_values, roi_histogram_values))

        self.canvas.draw()

    def update_histogram(self, all_histogram_values, roi_histogram_values):
        """Aktualisiert das Histogramm mit der ausgewählten Hintergrundfarbe."""
        bg_color = 'white' if self.bg_white_radio.isChecked() else 'black'
        self.figure.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        self.ax.cla() #Vorheriges Histogramm löschen

        self.ax.hist(all_histogram_values, bins=256, color='blue', alpha=0.5, label="Gesamt")
        if roi_histogram_values is not None:
            self.ax.hist(roi_histogram_values, bins=256, color='red', alpha=0.7, label="ROI")

        self.ax.set_title("Histogramm der Intensitätswerte", color=('white' if bg_color == 'black' else 'black')) 
        self.ax.set_xlabel("Intensitätswert", color=('white' if bg_color == 'black' else 'black')) 
        self.ax.set_ylabel("Häufigkeit", color=('white' if bg_color == 'black' else 'black')) 
        self.ax.tick_params(axis='x', colors=('white' if bg_color == 'black' else 'black')) 
        self.ax.tick_params(axis='y', colors=('white' if bg_color == 'black' else 'black')) 
        self.ax.spines['bottom'].set_color(('white' if bg_color == 'black' else 'black')) 
        self.ax.spines['top'].set_color(('white' if bg_color == 'black' else 'black'))    
        self.ax.spines['left'].set_color(('white' if bg_color == 'black' else 'black'))   
        self.ax.spines['right'].set_color(('white' if bg_color == 'black' else 'black'))  
        self.ax.grid(True, color=('grey' if bg_color == 'black' else 'lightgrey')) 
        self.ax.legend(labelcolor=('white' if bg_color == 'black' else 'black')) 
        self.canvas.draw()    

class VisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medizinische Visualisierung")
        self.setGeometry(100, 100, 800, 600)

        # Layout und zentrale Widgets
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Modus-Auswahl
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems(["Wähle Modus", "Student", "Doktor"])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_selected)
        self.layout.addWidget(self.mode_selector)

        # Render-Widget
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.layout.addWidget(self.vtk_widget)

        # VTK Renderer und Interactor
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        # Farben für die Regionen
        self.region_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  

        # Buttons
        self.load_button = QPushButton("Lade Daten")
        self.layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_data)

        self.unload_button = QPushButton("Entlade Daten")
        self.layout.addWidget(self.unload_button)
        self.unload_button.clicked.connect(self.unload_data)
        self.unload_button.hide()  # Standardmäßig ausgeblendet

        self.histogram_button = QPushButton("Histogramm anzeigen")
        self.histogram_button.clicked.connect(self.show_histogram)
        self.layout.addWidget(self.histogram_button)
        self.histogram_button.hide()  # Standardmäßig ausgeblendet

        # Slider für Slice-Steuerung
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.update_slice)
        self.layout.addWidget(QLabel("Slice-Wähler:"))
        self.layout.addWidget(self.slice_slider)
        self.slice_slider.hide()  # Standardmäßig ausgeblendet

        # Dropdown für Farbschema
        self.color_selector = QComboBox(self)
        self.color_selector.addItems(["Standard", "Graustufen", "Heiß/Kalt"])
        self.color_selector.currentIndexChanged.connect(self.update_color_map)
        self.layout.addWidget(QLabel("Farbschema:"))
        self.layout.addWidget(self.color_selector)
        self.color_selector.hide()  # Standardmäßig ausgeblendet

        # ROI-Button
        self.roi_button = QPushButton("ROI markieren")
        self.roi_button.clicked.connect(self.enable_roi_selection)
        self.layout.addWidget(self.roi_button)
        self.roi_button.hide()  # Standardmäßig ausgeblendet

        # Button zum Ein-/Ausblenden von Labels
        self.label_toggle_button = QPushButton("Labels ein/ausblenden")
        self.label_toggle_button.clicked.connect(self.toggle_labels)
        self.layout.addWidget(self.label_toggle_button)
        self.label_toggle_button.hide()  # Standardmäßig ausgeblendet

        # Legenden-Button
        self.legend_button = QPushButton("Legende ein/ausblenden")
        self.legend_button.clicked.connect(self.toggle_legend)
        self.layout.addWidget(self.legend_button)
        self.legend_button.hide()  # Standardmäßig ausgeblendet

        # Initialisiere VTK-Interactor
        self.interactor.Initialize()

        # Volumen-Daten
        self.volume = None
        self.slice_widget = None
        self.reader = None
        self.text_actors = []  # Liste für die Region-Beschriftungen
        self.legend_labels = []  # Liste für Legenden-Beschreibungen

        # ROI-Tools
        self.roi_widget = vtk.vtkBoxWidget()
        self.roi_enabled = False

        # Histogrammdaten
        self.histogram_values = []

        # Defining the annotations attribute
        self.annotations = [
            {"name": "Region 1", "position": (50, 50, 50), "description": "Lunge: Sauerstoffaufnahme und Gasaustausch."},
            {"name": "Region 2", "position": (100, 100, 100), "description": "Herz: Blutpumpe für den Körper."},
            {"name": "Region 3", "position": (160, 150, 150), "description": "Rippen: Schutz des Brustkorbs und der Organe."},
        ]

    def on_mode_selected(self, index):
        """Funktion zum Aktualisieren der UI basierend auf dem ausgewählten Modus."""
        if index == 1:  # Student-Modus
            self.renderer.SetBackground(0.5, 0.7, 1.0)  # Hintergrundfarbe für Student
            self.slice_slider.show()  # Slice-Steuerung im Student-Modus anzeigen
            self.color_selector.show()
            self.label_toggle_button.show()
            self.legend_button.show()  # Legende im Student-Modus anzeigen
            self.create_legend()  # Legende erstellen
            self.hide_all_mode_specific_widgets()  # Alle Widgets ausblenden
        elif index == 2:  # Doktor-Modus
            self.renderer.SetBackground(1.0, 0.8, 0.7)  # Hintergrundfarbe für Doktor
            self.histogram_button.show()  # Histogramm im Doktor-Modus anzeigen
            self.slice_slider.show()  # Slice-Steuerung im Doktor-Modus anzeigen
            self.color_selector.show()  # Farbauswahl im Doktor-Modus anzeigen
            self.roi_button.show()  # ROI im Doktor-Modus anzeigen
            self.label_toggle_button.show()  # Labels-Toggle-Button im Doktor-Modus anzeigen
            self.legend_button.hide()  # Legende im Doktor-Modus ausblenden
        else:
            self.renderer.SetBackground(0.5, 0.5, 0.5)  # Hintergrundfarbe für keinen Modus
            self.slice_slider.hide()
            self.color_selector.hide()
            self.label_toggle_button.hide()
            self.legend_button.hide()  # Legende ausblenden
            self.hide_all_mode_specific_widgets()  # Alle Widgets ausblenden

        self.vtk_widget.GetRenderWindow().Render()

    def create_legend(self):
        """Erstellt eine Legende mit den Region-Namen und den ausführlichen Beschreibungen."""
        legend_layout = QVBoxLayout()

        # Legende für jede Region erstellen
        for annotation in self.annotations:
            label = QLabel(annotation["name"])
            label.mousePressEvent = lambda event, ann=annotation: self.show_description(ann)
            legend_layout.addWidget(label)
            self.legend_labels.append(label)

        # Einen Container für die Legende erstellen
        legend_widget = QWidget()
        legend_widget.setLayout(legend_layout)

        self.layout.addWidget(legend_widget)
        legend_widget.hide()  # Standardmäßig ausgeblendet, wird durch Button sichtbar gemacht

        # Speichern des Widgets für spätere Steuerung
        self.legend_widget = legend_widget

    def toggle_legend(self):
        """Schaltet die Sichtbarkeit der Legende ein/aus."""
        # Umschalten der Sichtbarkeit der Legende
        if self.legend_widget.isVisible():
            self.legend_widget.hide()
        else:
            self.legend_widget.show()

        self.vtk_widget.GetRenderWindow().Render()

    def hide_all_mode_specific_widgets(self):
        """Versteckt alle spezifischen Widgets (für Student und Doktor-Modus)."""
        # self.slice_slider.hide()
        # self.color_selector.hide()
        self.roi_button.hide()
        self.histogram_button.hide()
        #self.label_toggle_button.hide()

    def load_data(self):
        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName("coronacases_org_004.vti")
        self.reader.Update()

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputConnection(self.reader.GetOutputPort())

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volume_mapper)
        self.volume.GetProperty().SetColor(self.get_color_transfer_function("Standard"))
        self.volume.GetProperty().SetScalarOpacity(self.get_opacity_transfer_function())  # Hier wird die Opazität gesetzt

        self.renderer.AddVolume(self.volume)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        # Weitere Initialisierungen...
        self.initialize_slice_viewer()
        self.load_button.setEnabled(False)
        self.unload_button.show()

        # 3D-Beschriftungen hinzufügen
        self.add_3d_labels()

        # Histogrammdaten generieren
        self.calculate_histogram()

    def unload_data(self):
        if self.volume:
            self.renderer.RemoveVolume(self.volume)
            self.volume = None

        if self.slice_widget:
            self.slice_widget.Off()
            self.slice_widget = None

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        self.load_button.setEnabled(True)
        self.unload_button.hide()
        self.roi_button.hide()
        self.slice_slider.hide()
        self.color_selector.hide()
        self.histogram_button.hide()

    def initialize_slice_viewer(self):
        self.slice_widget = vtk.vtkImagePlaneWidget()
        self.slice_widget.SetInteractor(self.interactor)
        self.slice_widget.SetInputConnection(self.reader.GetOutputPort())
        self.slice_widget.SetPlaneOrientationToZAxes()
        self.slice_widget.SetSliceIndex(50)
        self.slice_widget.DisplayTextOn()
        self.slice_widget.On()

        extent = self.reader.GetOutput().GetExtent()
        self.slice_slider.setMinimum(extent[4])
        self.slice_slider.setMaximum(extent[5])
        self.slice_slider.setValue(50)

    def update_slice(self, value):
        if self.slice_widget:
            self.slice_widget.SetSliceIndex(value)
            self.vtk_widget.GetRenderWindow().Render()

    def update_color_map(self, index):
        color_map = ["Standard", "Graustufen", "Heiß/Kalt"]
        if self.volume:
            self.volume.GetProperty().SetColor(self.get_color_transfer_function(color_map[index]))
            self.vtk_widget.GetRenderWindow().Render()

    def enable_roi_selection(self):
        if not self.roi_enabled:
            self.roi_widget.SetInteractor(self.interactor)
            self.roi_widget.SetPlaceFactor(1.0)
            self.roi_widget.SetInputData(self.reader.GetOutput())
            self.roi_widget.PlaceWidget()
            self.roi_widget.On()
            self.roi_enabled = True
            self.roi_button.setText("ROI deaktivieren")
            self.roi_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.roi_interaction_ended) # Observer hinzufügen
        else:
            self.roi_widget.Off()
            self.roi_enabled = False
            self.roi_button.setText("ROI markieren")


    def roi_interaction_ended(self, obj, event):
        """Wird aufgerufen, wenn die ROI-Interaktion beendet ist."""
        if self.volume:
            self.calculate_roi_histogram()

    def calculate_roi_histogram(self):
        """Berechnet das Histogramm der Intensitätswerte innerhalb der ROI."""

        if self.volume and self.roi_widget.GetEnabled(): # Überprüfe ob das Volumen geladen und der Widget aktiv ist
            polydata = vtk.vtkPolyData()
            self.roi_widget.GetPolyData(polydata)

            if polydata.GetNumberOfPoints() == 0:
                QMessageBox.warning(self,"Keine ROI ausgewählt", "Bitte wählen Sie eine ROI aus, die Daten enthält.")
                return

            bounds = polydata.GetBounds()

            image_data = self.reader.GetOutput()

            # Extrahiere die ROI-Daten
            extract = vtk.vtkExtractVOI()
            extract.SetInputData(image_data)
            extract.SetVOI(int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]), int(bounds[4]), int(bounds[5]))
            extract.Update()

            if extract.GetOutput().GetNumberOfPoints() == 0:
                QMessageBox.warning(self,"Keine ROI ausgewählt", "Bitte wählen Sie eine ROI aus, die Daten enthält.")
                return

            roi_voxel_data = extract_voxel_data(extract.GetOutput())
            self.roi_histogram_values = roi_voxel_data.flatten()
            self.show_histogram()
        else:
            QMessageBox.information(self, "Kein Volumen geladen", "Bitte laden Sie zuerst ein Volumen.")                

    def calculate_histogram(self):
        """Berechnet das Histogramm der gesamten Intensitätswerte."""
        image_data = self.reader.GetOutput()
        voxel_data = extract_voxel_data(image_data)
        self.histogram_values = voxel_data.flatten()
        self.roi_histogram_values = None #Zurücksetzen der ROI-Werte

    def show_histogram(self):
        """Zeigt das Histogramm mit optionalen ROI-Daten an."""
        dialog = HistogramDialog(self.histogram_values, self.roi_histogram_values)
        dialog.exec_()
    
    def get_color_transfer_function(self, color_map_name):
        """Gibt die Farbtransferschemen basierend auf dem Namen zurück."""
        if color_map_name == "Graustufen":
            color_transfer_function = vtk.vtkColorTransferFunction()
            color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
            color_transfer_function.AddRGBPoint(255, 1.0, 1.0, 1.0)
            return color_transfer_function
        elif color_map_name == "Heiß/Kalt":
            color_transfer_function = vtk.vtkColorTransferFunction()
            color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)  # Blau
            color_transfer_function.AddRGBPoint(500, 1.0, 0.5, 0.3) 

            return color_transfer_function
        else:  # Standard-Farbschema
            color_transfer_function = vtk.vtkColorTransferFunction()
            color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
            color_transfer_function.AddRGBPoint(255, 0.0, 1.0, 1.0)

            return color_transfer_function

    def get_opacity_transfer_function(self):
        """Gibt die Opazitäts-Transferfunktion zurück."""
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)   # Niedrigste Intensität: vollständig transparent
        opacity_transfer_function.AddPoint(255, 1.0) # Höchste Intensität: vollständig undurchsichtig
        return opacity_transfer_function

    def add_3d_labels(self):
        """Zeigt 3D-Labels mit farbigen Ecken an."""
        for i, annotation in enumerate(self.annotations):
            # Text-Actor
            text_actor = vtk.vtkBillboardTextActor3D()
            text_actor.SetInput(annotation["name"])
            text_actor.SetPosition(annotation["position"])
            text_actor.GetTextProperty().SetColor(1, 1, 1)  # Weißer Text
            text_actor.GetTextProperty().SetFontSize(12)
            self.renderer.AddActor(text_actor)
            self.text_actors.append(text_actor)

            # Farbige Ecke (vtkCubeSource für eine einfache Ecke)
            cube_source = vtk.vtkCubeSource()
            cube_source.SetXLength(5)  # Größe der Ecke anpassen
            cube_source.SetYLength(5)
            cube_source.SetZLength(5)
            cube_source.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cube_source.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.region_colors[i % len(self.region_colors)])  # Farbe aus der Liste
            actor.SetPosition(annotation["position"][0]-7, annotation["position"][1]-7, annotation["position"][2]-7) # Ecke etwas versetzen, damit sie neben dem Text ist.

            self.renderer.AddActor(actor)
            self.text_actors.append(actor) # Füge auch die Ecke der Liste hinzu, für das toggle

        self.vtk_widget.GetRenderWindow().Render()

    def toggle_labels(self):
        """Schaltet die Sichtbarkeit der 3D-Beschriftungen ein/aus."""
        for actor in self.text_actors:
            actor.SetVisibility(not actor.GetVisibility())
        self.vtk_widget.GetRenderWindow().Render()


    def show_description(self, annotation):
        """Zeigt die ausführliche Beschreibung der Region an."""
        description_dialog = QDialog(self)
        description_dialog.setWindowTitle(f"Beschreibung: {annotation['name']}")
        description_label = QLabel(annotation["description"], description_dialog)
        description_layout = QVBoxLayout(description_dialog)
        description_layout.addWidget(description_label)
        description_dialog.exec_()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationApp()
    window.show()
    app.exec_()
