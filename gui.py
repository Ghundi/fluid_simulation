#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluid‑simulation visualiser – PyQt6 version (3‑D data)

The script reads the binary files produced by the 3‑D C++ solver
(data.bin, v_x.bin, v_y.bin, v_z.bin, obs.bin) and shows them
in a Qt window with:
    • a time‑axis slider,
    • a Z‑slice slider,
    • a combo‑box to pick the visualised field (density / velocity‑X /
      velocity‑Y / velocity‑Z) and
    • an optional check‑box to draw velocity vectors (X/Y) on top of the
      density field.
"""

# ----------------------------------------------------------------------
# ── Imports
# ----------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PyQt6 import QtCore, QtGui, QtWidgets

# ----------------------------------------------------------------------
# ── Global configuration (must match the C++ simulation)
# ----------------------------------------------------------------------
# The C++ code stores the *padded* grid (width+2)*(height+2)*(depth+2)
# per time step, so we include the padding here as well.
width  = 128 + 2          # X‑size (including the two walls)
height = 64 + 2          # Y‑size (including the two walls)
depth  = 64  + 2          # Z‑size (including the two walls) – change if you
                          # compiled the C++ solver with a different depth

# Custom colour map for density (white → green → blue → red)
density_cmap = LinearSegmentedColormap.from_list(
    "density_cmap",
    ["white", "lightgreen", "green", "deepskyblue", "blue", "darkred", "red"],
)

# ----------------------------------------------------------------------
# ── Helper functions
# ----------------------------------------------------------------------
def np_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """
    Convert an (H, W, 3) uint8 NumPy array to a QImage.
    In PyQt6 the format enum lives under QtGui.QImage.Format.
    """
    h, w, ch = arr.shape
    assert ch == 3
    img = QtGui.QImage(
        arr.tobytes(), w, h,
        3 * w,                     # bytes per line
        QtGui.QImage.Format.Format_RGB888
    ).copy()
    return img


def apply_cmap(data_2d: np.ndarray,
               cmap: LinearSegmentedColormap,
               vmin: float,
               vmax: float) -> np.ndarray:
    """Map a 2‑D float array to an (H, W, 3) uint8 RGB image."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(data_2d))               # (H, W, 4) floats in [0, 1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)   # drop alpha
    return rgb


def overlay_obstacle(base_rgb: np.ndarray,
                    obs_2d: np.ndarray,
                    alpha: float = 0.2) -> np.ndarray:
    """Darken pixels where the obstacle mask is 1."""
    mask = obs_2d > 0.5
    base_rgb[mask] = (base_rgb[mask].astype(np.float32) *
                      (1 - alpha)).astype(np.uint8)
    return base_rgb


def draw_vectors(pix: QtGui.QPixmap,
                 vx: np.ndarray,
                 vy: np.ndarray,
                 skip: int = 30,
                 scale: float = 0.2,
                 color: QtGui.QColor = QtGui.QColor(255, 255, 0),
                 thickness: int = 1) -> QtGui.QPixmap:
    """
    Paint X/Y velocity vectors onto a QPixmap using QPainter.
    """
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(color)
    pen.setWidth(thickness)
    painter.setPen(pen)

    h, w = vx.shape                     # (rows, cols) = (y, x)

    head_len   = 6
    head_angle = np.radians(30)

    for y in range(skip // 2, h, skip):
        for x in range(skip // 2, w, skip):
            u, v = vx[y, x], vy[y, x]
            if np.hypot(u, v) < 0.02:
                continue

            end_x = x + u * scale
            end_y = y + v * scale

            painter.drawLine(QtCore.QPointF(x, y),
                             QtCore.QPointF(end_x, end_y))

            theta = np.arctan2(v, u)
            for sign in (+1, -1):
                hx = end_x - head_len * np.cos(theta + sign * head_angle)
                hy = end_y - head_len * np.sin(theta + sign * head_angle)
                painter.drawLine(QtCore.QPointF(end_x, end_y),
                                 QtCore.QPointF(hx, hy))

    painter.end()
    return pix

# ----------------------------------------------------------------------
# ── Main Window (Qt)
# ----------------------------------------------------------------------
class FluidViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fluid‑simulation visualiser (3‑D)")
        self.resize(1000, 700)

        self.depth = depth

        # --------------------------------------------------------------
        # 1️⃣ Load all binary files (once)
        # --------------------------------------------------------------
        self.load_data()

        # --------------------------------------------------------------
        # 2️⃣ Build the GUI
        # --------------------------------------------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # ----- image display ------------------------------------------------
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        vbox.addWidget(self.image_label, 1)

        # ----- control panel ------------------------------------------------
        ctrl = QtWidgets.QHBoxLayout()
        vbox.addLayout(ctrl)

        # ---- time slider --------------------------------------------------
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.n_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow
        )
        self.time_slider.setTickInterval(max(1, self.n_frames // 10))
        self.time_slider.valueChanged.connect(self.update_image)

        ctrl.addWidget(QtWidgets.QLabel("Frame:"))
        ctrl.addWidget(self.time_slider, stretch=1)

        # ---- Z‑slice slider ------------------------------------------------
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.depth - 1)
        self.slice_slider.setValue(self.depth // 2)   # start in the middle
        self.slice_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow
        )
        self.slice_slider.setTickInterval(max(1, self.depth // 10))
        self.slice_slider.valueChanged.connect(self.update_image)

        ctrl.addWidget(QtWidgets.QLabel("Slice:"))
        ctrl.addWidget(self.slice_slider, stretch=1)

        # ---- field selector ------------------------------------------------
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.addItems(
            ["Density", "Velocity X", "Velocity Y", "Velocity Z"]
        )
        self.field_combo.currentIndexChanged.connect(self.update_image)
        ctrl.addWidget(self.field_combo)

        # ---- vectors check‑box ---------------------------------------------
        self.vec_checkbox = QtWidgets.QCheckBox("Show vectors")
        self.vec_checkbox.setChecked(True)
        self.vec_checkbox.toggled.connect(self.update_image)
        ctrl.addWidget(self.vec_checkbox)

        # ----- status bar ---------------------------------------------------
        self.status = self.statusBar()
        self.update_status()

        # --------------------------------------------------------------
        # 3️⃣ Show the first frame
        # --------------------------------------------------------------
        self.update_image()

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
            with open(path, "rb") as f:
                arr = np.fromfile(f, dtype=np.float32)

            # ------------------------------------------------------------------
            # sanity check – total number of floats must be a multiple of a frame
            # ------------------------------------------------------------------
            frame_elems = width * height * depth
            assert arr.size % frame_elems == 0, f"bad size in {name}"
            # reshape to (time, z, y, x)
            return arr.reshape(-1, depth, height, width)

        # ------------------------------------------------------------------
        # Load the five fields
        # ------------------------------------------------------------------
        self.density = read_bin("data.bin")
        self.vx      = read_bin("v_x.bin")
        self.vy      = read_bin("v_y.bin")
        self.vz      = read_bin("v_z.bin")
        self.obs     = read_bin("obs.bin")

        self.n_frames = self.density.shape[0]

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_status(self):
        """Refresh the text shown in the status bar."""
        self.status.showMessage(
            f"frame {self.time_slider.value() + 1}/{self.n_frames}   "
            f"slice {self.slice_slider.value()}/{self.depth - 1}"
        )

    # ------------------------------------------------------------------
    # Core rendering routine
    # ------------------------------------------------------------------
    def update_image(self):
        """
        Re‑create the displayed image whenever any control changes.
        The routine extracts the requested Z‑slice, colour‑maps it and,
        if requested, draws X/Y velocity vectors on top of the density.
        """
        frame_idx = self.time_slider.value()
        slice_idx = self.slice_slider.value()
        field = self.field_combo.currentText()
        show_vec = self.vec_checkbox.isChecked()

        # --------------------------------------------------------------
        # 1️⃣ pick the 2‑D slice that we want to visualise
        # --------------------------------------------------------------
        if field == "Density":
            raw = self.density[frame_idx, slice_idx, :, :]
            vmin, vmax = 0.0, 0.01
            rgb = apply_cmap(raw, density_cmap, vmin, vmax)

        elif field == "Velocity X":
            raw = self.vx[frame_idx, slice_idx, :, :]
            vmin, vmax = -10.0, 10.0
            rgb = apply_cmap(raw, density_cmap, vmin, vmax)   # same colour map

        elif field == "Velocity Y":
            raw = self.vy[frame_idx, slice_idx, :, :]
            vmin, vmax = -1.0, 1.0
            rgb = apply_cmap(raw, density_cmap, vmin, vmax)

        else:   # "Velocity Z"
            raw = self.vz[frame_idx, slice_idx, :, :]
            vmin, vmax = -1.0, 1.0
            rgb = apply_cmap(raw, density_cmap, vmin, vmax)

        # --------------------------------------------------------------
        # 2️⃣ overlay obstacles (they are stored as 0/1 floats)
        # --------------------------------------------------------------
        obs_slice = self.obs[frame_idx, slice_idx, :, :]
        rgb = overlay_obstacle(rgb, obs_slice, alpha=0.2)

        # --------------------------------------------------------------
        # 3️⃣ Convert to QPixmap
        # --------------------------------------------------------------
        qimg = np_to_qimage(rgb)
        pix = QtGui.QPixmap.fromImage(qimg)

        # --------------------------------------------------------------
        # 4️⃣ Optional velocity vectors – only sensible on density view
        # --------------------------------------------------------------
        if show_vec and field == "Density":
            vx_slice = self.vx[frame_idx, slice_idx, :, :]
            vy_slice = self.vy[frame_idx, slice_idx, :, :]
            pix = draw_vectors(
                pix,
                vx_slice,
                vy_slice,
                skip=30,
                scale=0.2,
                color=QtGui.QColor(255, 255, 0),
                thickness=1,
            )

        # --------------------------------------------------------------
        # 5️⃣ Scale the pixmap to fit the label while keeping aspect ratio
        # --------------------------------------------------------------
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

        # --------------------------------------------------------------
        # 6️⃣ Update status bar
        # --------------------------------------------------------------
        self.update_status()

    # ------------------------------------------------------------------
    # Resize handling – keep image centred and scaled
    # ------------------------------------------------------------------
    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.update_image()   # redraw with the new widget size


# ----------------------------------------------------------------------
# ── Application entry point
# ----------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = FluidViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()