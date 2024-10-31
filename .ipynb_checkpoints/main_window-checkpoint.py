# main_window.py
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSpacerItem, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from modeling import kmeans_clustering, linear_regression  # Updated import


# Matplotlib Canvas for PyQt6
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots()
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Olympic Success Predictor")
        layout = QVBoxLayout()

        # Tabs for clustering and predictions
        tabs = QTabWidget()
        tabs.addTab(self.create_clustering_tab(), "K-Means Clusters")
        tabs.addTab(self.create_prediction_tab(), "Medal Prediction")
        layout.addWidget(tabs)

        self.setLayout(layout)

    def create_clustering_tab(self):
        # Create the K-means Clustering tab
        tab = QWidget()
        layout = QVBoxLayout()

        # Generate K-means clusters and create scatter plot
        df = kmeans_clustering()  # Get the clustered data
        canvas = MplCanvas(self)
        sns.scatterplot(data=df, x='gdp', y='total', hue='Cluster', ax=canvas.ax)
        canvas.ax.set_title('K-means Clusters (GDP vs. Total Medals)')
        canvas.ax.set_xlabel('GDP')
        canvas.ax.set_ylabel('Total Medals')

        layout.addWidget(canvas)
        tab.setLayout(layout)
        return tab

    def create_prediction_tab(self):
        # Create the Medal Prediction tab
        tab = QWidget()
        layout = QVBoxLayout()

        # Input section for GDP
        input_layout = QHBoxLayout()
        gdp_label = QLabel("Enter GDP per capita for Prediction:")
        self.gdp_input = QLineEdit(self)
        self.gdp_input.setPlaceholderText("Type GDP here...")
        self.gdp_input.setMaximumWidth(200)

        input_layout.addWidget(gdp_label)
        input_layout.addWidget(self.gdp_input)
        layout.addLayout(input_layout)

        # Add spacing for readability
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Prediction output label
        self.result_label = QLabel("Prediction Result:")
        self.result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.result_label)

        # Prediction button
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(predict_button)

        tab.setLayout(layout)
        return tab

    def make_prediction(self):
        # Perform linear regression prediction based on user input
        try:
            # Retrieve GDP input
            gdp_value = float(self.gdp_input.text())
            model, _ = linear_regression()  # Load the trained linear regression model

            # Predict medal count without normalization
            predicted_medals = model.predict([[gdp_value]])

            # Display result
            self.result_label.setText(f"Predicted Medal Count: {predicted_medals[0]:.2f}")
        except ValueError:
            self.result_label.setText("Invalid GDP value entered")
        except Exception as e:
            self.result_label.setText(f"Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
