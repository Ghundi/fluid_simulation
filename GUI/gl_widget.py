"""
OpenGL 3D View widget for the fluid simulation visualizer.
"""

import OpenGL.GL as gl
import numpy as np
import OpenGL.GLU as glu
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6 import QtCore, QtGui, QtWidgets

import config

class OpenGL3DView(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

        # Apply the same transform to everything drawn from CFD index space
        self.domain_offset = (-1.0, -1.0, -1.0)  # matches your grid/box offsets
        self.obstacle_ibo = None
        self.obstacle_index_count = 0

        
        # Camera parameters
        self.camera_distance = 200
        self.camera_azimuth = 45
        self.camera_elevation = 30
        self.camera_x = 0
        self.camera_y = 0
        self.is_panning = False
        self.last_mouse_pos = None
        self.is_rotating = False
        
        # Visualization data
        self.obstacle_verts = []
        self.obstacle_faces = []
        self.streamlines = []
        self.streamline_colors = []
        self.obstacle_vbo = None
        self.obstacle_faces_count = 0
        self.streamline_vbo = None
        self.streamline_count = 0
        self.streamline_indices = []
        
        # Grid parameters
        self.grid_size = 5
        self.grid_width = config.width
        self.grid_height = config.height
        self.grid_depth = config.depth
    
    def initializeGL(self):
        """Initialize OpenGL resources and state."""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
    
    def resizeGL(self, w, h):
        """Respond to window resizing."""
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, w / h if h > 0 else 1, 0.1, 1000)
        gl.glMatrixMode(gl.GL_MODELVIEW)
    
    def paintGL(self):
        """Paint the scene."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        
        # Set up camera
        gl.glTranslatef(0, 0, -self.camera_distance)
        gl.glRotatef(self.camera_elevation, 1, 0, 0)
        gl.glRotatef(self.camera_azimuth, 0, 1, 0)
        gl.glTranslatef(self.camera_x, self.camera_y, 0)
        
        # Draw grid (shifted by -1 to align with CFD data origin)
        self.draw_grid(-1, -1, -1)
        
        # Draw coordinate axes (shifted by -1)
        self.draw_axes(-1, -1, -1)
        
        # Draw bounding box
        self.draw_bounding_box()
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw streamlines
        self.draw_streamlines()
    
    def draw_grid(self, x_offset, y_offset, z_offset):
        """Draw a 3D grid."""
        gl.glColor4f(0.3, 0.3, 0.3, 0.5)
        gl.glLineWidth(1.0)
        
        # X-Y plane grid
        gl.glBegin(gl.GL_LINES)
        for x in range(0, self.grid_width, self.grid_size):
            gl.glVertex3f(x + x_offset, 0 + y_offset, 0 + z_offset)
            gl.glVertex3f(x + x_offset, self.grid_height + y_offset, 0 + z_offset)
            
            gl.glVertex3f(0 + x_offset, x + y_offset, 0 + z_offset)
            gl.glVertex3f(self.grid_width + x_offset, x + y_offset, 0 + z_offset)
        
        # X-Z plane grid
        for x in range(0, self.grid_width, self.grid_size):
            gl.glVertex3f(x + x_offset, 0 + y_offset, 0 + z_offset)
            gl.glVertex3f(x + x_offset, 0 + y_offset, self.grid_depth + z_offset)
            
            gl.glVertex3f(0 + x_offset, 0 + y_offset, x + z_offset)
            gl.glVertex3f(self.grid_width + x_offset, 0 + y_offset, x + z_offset)
        
        # Y-Z plane grid
        for y in range(0, self.grid_height, self.grid_size):
            gl.glVertex3f(0 + x_offset, y + y_offset, 0 + z_offset)
            gl.glVertex3f(0 + x_offset, y + y_offset, self.grid_depth + z_offset)
            
            gl.glVertex3f(0 + x_offset, 0 + y_offset, y + z_offset)
            gl.glVertex3f(self.grid_width + x_offset, 0 + y_offset, y + z_offset)
        gl.glEnd()
    
    def draw_axes(self, x_offset, y_offset, z_offset):
        """Draw coordinate axes."""
        # X-axis (red)
        gl.glColor4f(1.0, 0.0, 0.0, 1.0)
        gl.glLineWidth(2.5)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(x_offset, y_offset, z_offset)
        gl.glVertex3f(x_offset + 20, y_offset, z_offset)
        gl.glEnd()
        
        # Y-axis (green)
        gl.glColor4f(0.0, 1.0, 0.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(x_offset, y_offset, z_offset)
        gl.glVertex3f(x_offset, y_offset + 20, z_offset)
        gl.glEnd()
        
        # Z-axis (blue)
        gl.glColor4f(0.0, 0.0, 1.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(x_offset, y_offset, z_offset)
        gl.glVertex3f(x_offset, y_offset, z_offset + 20)
        gl.glEnd()
    
    def draw_bounding_box(self):
        """Draw a bounding box around the simulation domain."""
        gl.glColor4f(1.0, 1.0, 1.0, 0.3)
        gl.glLineWidth(1.5)
        
        x0, y0, z0 = -1, -1, -1
        x1, y1, z1 = config.width-1, config.height-1, config.depth-1
        
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex3f(x0, y0, z0)
        gl.glVertex3f(x1, y0, z0)
        gl.glVertex3f(x1, y1, z0)
        gl.glVertex3f(x0, y1, z0)
        gl.glEnd()
        
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex3f(x0, y0, z1)
        gl.glVertex3f(x1, y0, z1)
        gl.glVertex3f(x1, y1, z1)
        gl.glVertex3f(x0, y1, z1)
        gl.glEnd()
        
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(x0, y0, z0)
        gl.glVertex3f(x0, y0, z1)
        
        gl.glVertex3f(x1, y0, z0)
        gl.glVertex3f(x1, y0, z1)
        
        gl.glVertex3f(x1, y1, z0)
        gl.glVertex3f(x1, y1, z1)
        
        gl.glVertex3f(x0, y1, z0)
        gl.glVertex3f(x0, y1, z1)
        gl.glEnd()
    
    def draw_obstacles(self):
        """Draw obstacles with solid fill + wireframe overlay."""
        if not self.obstacle_verts or not self.obstacle_faces:
            return
        if self.obstacle_vbo is None or self.obstacle_ibo is None:
            self._create_obstacle_vbo()

        self._push_domain()

        # Enable polygon offset to avoid z-fighting
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1.0, 1.0)

        # Draw solid faces
        gl.glColor4f(0.5, 0.5, 0.5, 1.0)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        self._draw_obstacle_triangles()

        # Draw wireframe edges
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
        gl.glPolygonOffset(1.1, 1.1)
        gl.glColor4f(0.0, 0.0, 0.0, 1.0)  # Black edges
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(1.5)
        self._draw_obstacle_triangles()

        # Reset
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)

        self._pop_domain()

    def _draw_obstacle_triangles(self):
        """Helper to draw obstacle triangles (used for both fill and wireframe)."""
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.obstacle_vbo)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.obstacle_ibo)
        gl.glDrawElements(gl.GL_TRIANGLES, self.obstacle_index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


    def _create_obstacle_vbo(self):
        """Create VBO + IBO for obstacle rendering (indexed triangles)."""
        if not self.obstacle_verts or not self.obstacle_faces:
            return

        vertices = np.asarray(self.obstacle_verts, dtype=np.float32)  # (N, 3)
        indices  = np.asarray(self.obstacle_faces, dtype=np.uint32).ravel()  # (M*3,)

        # Vertex buffer
        self.obstacle_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.obstacle_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # Index buffer
        self.obstacle_ibo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.obstacle_ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        self.obstacle_index_count = indices.size

        
    def draw_streamlines(self):
        """Draw streamlines using a simple line strip."""
        if not self.streamlines:
            return

        # This is the bug fix: Apply the same transformation used by the obstacles
        # so the streamlines are drawn in the correct coordinate space.
        self._push_domain()

        gl.glLineWidth(1.5)
        for idx, streamline in enumerate(self.streamlines):
            # Set the color for this specific streamline
            r, g, b, a = self.streamline_colors[idx]
            gl.glColor4f(r, g, b, a)

            # Draw the line
            gl.glBegin(gl.GL_LINE_STRIP)
            for pt in streamline:
                # pt is a numpy array [x, y, z]
                gl.glVertex3fv(pt)
            gl.glEnd()
        
        # Pop the transformation matrix to not affect subsequent drawing
        self._pop_domain()


    def _create_streamline_vbo(self):
        """Create VBO for streamline rendering."""
        if not self.streamlines:
            return
        
        # Flatten all streamlines into a single array
        all_points = []
        self.streamline_indices = []
        offset = 0
        
        for streamline in self.streamlines:
            all_points.extend(streamline)
            self.streamline_indices.append((offset, len(streamline)))
            offset += len(streamline)
        
        # Convert to numpy array
        vertices = np.array(all_points, dtype=np.float32)
        
        # Create VBO
        self.streamline_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.streamline_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        
        self.streamline_count = len(self.streamlines)
    
    def mousePressEvent(self, event):
        """Handle mouse press events for rotation."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_rotating = True
            self.last_mouse_pos = event.position()
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for rotation."""
        if self.is_rotating and self.last_mouse_pos is not None:
            # Calculate mouse movement
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()
            
            # Update camera angles
            self.camera_azimuth += dx * 0.5
            self.camera_elevation += dy * 0.5
            
            # Keep elevation in reasonable range
            self.camera_elevation = max(-89, min(89, self.camera_elevation))
            
            self.last_mouse_pos = event.position()
            self.update()
        elif self.is_panning and self.last_mouse_pos is not None:
            # Calculate mouse movement
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()
            
            # Update camera position
            self.camera_x += dx * 0.1
            self.camera_y -= dy * 0.1
            
            self.last_mouse_pos = event.position()
            self.update()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_rotating = False
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.is_panning = False
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        # Adjust zoom speed
        zoom_factor = 1.1 if event.angleDelta().y() < 0 else 0.9
        self.camera_distance *= zoom_factor
        self.camera_distance = max(10, min(500, self.camera_distance))
        self.update()
        super().wheelEvent(event)
    
    def set_obstacle_data(self, verts, faces):
        """Set obstacle data for rendering."""
        self.obstacle_verts = verts
        self.obstacle_faces = faces
        self.obstacle_vbo = None  # Force recreation of VBO
        self.update()
    
    def set_streamline_data(self, streamlines, colors):
        """Set streamline data for rendering."""
        self.streamlines = streamlines
        self.streamline_colors = colors
        self.streamline_vbo = None  # Force recreation of VBO
        self.update()

    def _push_domain(self):
        gl.glPushMatrix()
        gl.glTranslatef(*self.domain_offset)

    def _pop_domain(self):
        gl.glPopMatrix()