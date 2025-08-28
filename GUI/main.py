"""
Application entry point for the fluid simulation visualizer.
"""

import os
import sys
from PyQt6 import QtWidgets

from main_window import Fluid3DViewer

def main():
    # Check if required data files exist
    data_dir = "data"
    required_files = ["data.bin", "v_x.bin", "v_y.bin", "v_z.bin", "obs.bin"]
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please run the C++ simulation first to generate the data files.")
        return 1
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing_files:
        print(f"Error: Missing data files: {', '.join(missing_files)}")
        print("Please run the C++ simulation first to generate the data files.")
        return 1
    
    # Initialize and run the application
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    viewer = Fluid3DViewer()
    viewer.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()