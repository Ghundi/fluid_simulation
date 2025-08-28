"""
Main window for the fluid simulation visualizer.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
import time
import numpy as np
import os

import gl_widget
import utils
import config

class Fluid3DViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Fluid Simulation Visualizer (Pure Qt OpenGL)")
        self.resize(1200, 800)
        
        # --------------------------------------------------------------
        # 1️⃣ Load all binary files (once)
        # --------------------------------------------------------------
        self.load_data()
        
        # --------------------------------------------------------------
        # 2️⃣ Build the GUI
        # --------------------------------------------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # ----- 3D View ------------------------------------------------
        self.view = gl_widget.OpenGL3DView()
        main_layout.addWidget(self.view, 4)
        
        # ----- control panel ------------------------------------------
        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setMaximumWidth(300)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)
        
        # Time slider
        time_group = QtWidgets.QGroupBox("Time Control")
        time_layout = QtWidgets.QVBoxLayout()
        
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.n_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow
        )
        self.time_slider.setTickInterval(max(1, self.n_frames // 10))
        self.time_slider.valueChanged.connect(self.update_visualization)
        
        time_layout.addWidget(QtWidgets.QLabel("Frame:"))
        time_layout.addWidget(self.time_slider)
        time_group.setLayout(time_layout)
        ctrl_layout.addWidget(time_group)
        
        # Visualization options
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()
        
        # Obstacles
        self.obstacle_checkbox = QtWidgets.QCheckBox("Show Obstacles")
        self.obstacle_checkbox.setChecked(True)
        self.obstacle_checkbox.toggled.connect(self.update_visualization)
        vis_layout.addWidget(self.obstacle_checkbox)
        
        # Streamlines
        self.streamline_checkbox = QtWidgets.QCheckBox("Show Streamlines")
        self.streamline_checkbox.setChecked(True)
        self.streamline_checkbox.toggled.connect(self.update_visualization)
        vis_layout.addWidget(self.streamline_checkbox)

        # Proximity Streamline Visibility control
        streamline_proximity_layout = QtWidgets.QHBoxLayout()
        streamline_proximity_layout.addWidget(QtWidgets.QLabel("Streamline Proximity:"))
        self.streamline_proximity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.streamline_proximity_slider.setMinimum(1)
        self.streamline_proximity_slider.setMaximum(30)
        self.streamline_proximity_slider.setValue(config.STREAMLINE_PROXIMITY)
        self.streamline_proximity_slider.valueChanged.connect(self.update_streamline_params)
        streamline_proximity_layout.addWidget(self.streamline_proximity_slider)
        vis_layout.addLayout(streamline_proximity_layout)
        
        # Streamline density control
        streamline_density_layout = QtWidgets.QHBoxLayout()
        streamline_density_layout.addWidget(QtWidgets.QLabel("Streamline Density:"))
        self.streamline_density_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.streamline_density_slider.setMinimum(5)
        self.streamline_density_slider.setMaximum(30)
        self.streamline_density_slider.setValue(config.STREAMLINE_DENSITY)
        self.streamline_density_slider.valueChanged.connect(self.update_streamline_params)
        streamline_density_layout.addWidget(self.streamline_density_slider)
        vis_layout.addLayout(streamline_density_layout)
        
        # Streamline length control
        streamline_length_layout = QtWidgets.QHBoxLayout()
        streamline_length_layout.addWidget(QtWidgets.QLabel("Streamline Length:"))
        self.streamline_length_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.streamline_length_slider.setMinimum(100)
        self.streamline_length_slider.setMaximum(1000)
        self.streamline_length_slider.setValue(config.INTEGRATION_STEPS)
        self.streamline_length_slider.valueChanged.connect(self.update_streamline_params)
        streamline_length_layout.addWidget(self.streamline_length_slider)
        vis_layout.addLayout(streamline_length_layout)
        
        vis_group.setLayout(vis_layout)
        ctrl_layout.addWidget(vis_group)
        
        # Performance info
        perf_group = QtWidgets.QGroupBox("Performance")
        perf_layout = QtWidgets.QVBoxLayout()
        
        self.fps_label = QtWidgets.QLabel("FPS: --")
        perf_layout.addWidget(self.fps_label)
        
        self.render_time_label = QtWidgets.QLabel("Render time: -- ms")
        perf_layout.addWidget(self.render_time_label)
        
        perf_group.setLayout(perf_layout)
        ctrl_layout.addWidget(perf_group)
        
        # Status bar
        self.status = self.statusBar()
        # self.update_status()
        
        ctrl_layout.addStretch(1)
        main_layout.addWidget(ctrl_panel, 1)
        
        # --------------------------------------------------------------
        # 3️⃣ Initialize 3D visualization items
        # --------------------------------------------------------------
        self.last_update_time = time.time()
        self.update_visualization()
        
        # Setup a timer for FPS monitoring
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update FPS display once per second
    
    def update_fps(self):
        """Update FPS display in the performance panel."""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.last_update_time = current_time
    
    # def update_velocity_threshold(self):
    #     """Update the velocity change threshold from the slider."""
    #     config.VELOCITY_CHANGE_THRESHOLD = self.velocity_change_slider.value() / 100.0
    #     self.update_visualization()
    
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        """
        Read the five *.bin files into arrays of shape
        (n_frames, depth, height, width).  All files contain ``float32`` data.
        """
        def read_bin(name):
            path = os.path.join("data", name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
                
            with open(path, "rb") as f:
                arr = np.fromfile(f, dtype=np.float32)
            
            # Sanity check – total number of floats must be a multiple of a frame
            frame_elems = config.width * config.height * config.depth
            if arr.size % frame_elems != 0:
                raise ValueError(f"Bad size in {name}: {arr.size} elements, expected multiple of {frame_elems}")
            
            # reshape to (time, z, y, x)
            return arr.reshape(-1, config.depth, config.height, config.width)
        
        # Load the five fields
        self.density = read_bin("data.bin")
        self.vx = read_bin("v_x.bin")
        self.vy = read_bin("v_y.bin")
        self.vz = read_bin("v_z.bin")
        self.obs = read_bin("obs.bin")
        
        self.n_frames = self.density.shape[0]
        print(f"Loaded data with {self.n_frames} frames of size {config.width}x{config.height}x{config.depth}")
    # ------------------------------------------------------------------
    # Streamline parameter updates
    # ------------------------------------------------------------------
    def update_streamline_params(self):
        """Update streamline parameters from sliders."""
        config.STREAMLINE_PROXIMITY = self.streamline_proximity_slider.value()
        config.STREAMLINE_DENSITY = self.streamline_density_slider.value()
        config.INTEGRATION_STEPS = self.streamline_length_slider.value()
        
        # Update visualization with new parameters
        self.update_visualization()
    # ------------------------------------------------------------------
    # Core visualization routine
    # ------------------------------------------------------------------
    def update_visualization(self):
        """Update all visualization elements based on current settings."""
        start_time = time.time()
        
        frame_idx = self.time_slider.value()
        
        # Get obstacle data for current frame
        obs_data = self.obs[frame_idx, :, :, :]  # (z, y, x)
        
        # Reshape to (x, y, z) for visualization
        obs_data = np.transpose(obs_data, (2, 1, 0))  # (x, y, z)
        
        # Generate mesh
        mesh_data = utils.generate_obstacle_mesh(obs_data)
        
        # Prepare obstacle data for OpenGL
        obstacle_verts = []
        obstacle_faces = []
        
        # Set obstacle data
        if self.obstacle_checkbox.isChecked():
            if mesh_data['vertexes'].size > 0:
                # Convert to format OpenGL can use
                for i in range(len(mesh_data['vertexes'])):
                    x, y, z = mesh_data['vertexes'][i]
                    # Shift by -1 to align with CFD data origin
                    obstacle_verts.append([x-1, y-1, z-1])
                
                for face in mesh_data['faces']:
                    obstacle_faces.append([face[0], face[1], face[2]])
            
        self.view.set_obstacle_data(obstacle_verts, obstacle_faces)
        
        # Get streamline data
        streamlines = []
        streamline_colors = []
        
        if self.streamline_checkbox.isChecked():
            # Get velocity and obstacle data
            vx_data = self.vx[frame_idx, :, :, :]  # (z, y, x)
            vy_data = self.vy[frame_idx, :, :, :]  # (z, y, x)
            vz_data = self.vz[frame_idx, :, :, :]  # (z, y, x)
            obs_data = self.obs[frame_idx, :, :, :]  # (z, y, x)
            density_data = self.density[frame_idx, :, :, :] if True else None
            
            # Reshape to (x, y, z) for streamline generation
            vx = np.transpose(vx_data, (2, 1, 0))  # (x, y, z)
            vy = np.transpose(vy_data, (2, 1, 0))  # (x, y, z)
            vz = np.transpose(vz_data, (2, 1, 0))  # (x, y, z)
            obs = np.transpose(obs_data, (2, 1, 0))  # (x, y, z)
            density = np.transpose(density_data, (2, 1, 0)) if density_data is not None else None
            
            # Generate streamlines (only where velocity changes)
            streamlines, streamline_colors = utils.generate_streamlines(vx=vx, vy=vy, vz=vz, obs_data=obs)
            
            # Shift all points by -1 to align with CFD data origin
            for i in range(len(streamlines)):
                streamlines[i] = streamlines[i] - 1
        
        # Set streamline data
        self.view.set_streamline_data(streamlines, streamline_colors)
        
        # Update status
        # self.update_status()
        
        # Update render time
        render_time = (time.time() - start_time) * 1000
        self.render_time_label.setText(f"Render time: {render_time:.1f} ms")
    
    # ------------------------------------------------------------------
    # UI helpers
    # ----------------------------------------------------------------