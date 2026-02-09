# ising_viz.py

import os
# Hide pygame support prompt before importing
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import warnings
# Suppress runtime and user warnings (e.g., AVX2, pkg_resources)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import contextlib

# Suppress stdout/stderr temporarily during pygame import
with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
    import pygame
    pygame.init()
    from pygame.locals import *

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import math
import time
import argparse
from ising_initialization import resolve_open_dir

# ================================
# Visualization parameters
# ================================
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
VOXEL_SIZE = 1.0
COLOR_SPIN_UP = (0.8, 0.8, 0.8)
COLOR_SPIN_DOWN = (0.3, 0.5, 0.5)
COLOR_BACKGROUND = (0.1, 0.1, 0.1, 1.0)

PLAYBACK_FPS = 50  # how fast to advance snapshots per second

# ================================
# OpenGL visualizer
# ================================
class IsingVisualizer3D:
    def __init__(self, snapshots, metadata, voxel_size=VOXEL_SIZE):
        self.snapshots = snapshots
        self.metadata = metadata
        self.num_snapshots = len(snapshots)
        self.current_idx = 0
        self.voxel_size = voxel_size

        self.last_frame_time = time.time()
        self.playing = True
        self.reverse = False
        self.playback_fps = PLAYBACK_FPS  # can change with buttons

        self.H = snapshots[0]['spins'].shape[0]
        self.W = snapshots[0]['spins'].shape[1]
        self.L = snapshots[0]['spins'].shape[2]

        # Precompute surface indices
        self.surface_indices = [
            (i,j,k) for i in range(self.H)
                     for j in range(self.W)
                     for k in range(self.L)
                     if i==0 or i==self.H-1 or j==0 or j==self.W-1 or k==0 or k==self.L-1
        ]

        # Pygame + OpenGL init
        pygame.init()
        pygame.font.init()

        self.font = pygame.font.SysFont("Arial", 12)  # name, size
        # Pre-render static labels
        self.text_surfaces = {}
        for label in ["Magnetization", "-1", "+1"]:
            surf = self.font.render(label, True, (255, 255, 255))
            self.text_surfaces[label] = (surf, pygame.image.tostring(surf, "RGBA", True), surf.get_width(), surf.get_height())

        self.text_surfaces = {}

        # Precompute static labels
        for key in ["Step", "Energy", "Magnetization", "Temperature"]:
            surf = self.font.render(f"{key}: ", True, (255, 255, 255))
            data = pygame.image.tostring(surf, "RGBA", True)
            self.text_surfaces[key] = {
                "surf": surf,
                "data": data,
                "w": surf.get_width(),
                "h": surf.get_height()
            }

        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Ising Model Playback")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*COLOR_BACKGROUND)

        self.shader_program = self.compile_shaders()
        self.VAO, self.num_indices = self.create_cube_vao()
        self.instance_positions, self.instance_colors = self.get_surface_instances()
        self.update_instance_buffers()

    # --------------------
    # Shaders
    # --------------------
    def compile_shaders(self):
        vertex_src = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 iPos;
        layout (location = 2) in vec3 iColor;
        uniform mat4 MVP;
        out vec3 vColor;
        void main()
        {
            vec3 worldPos = aPos + iPos;
            gl_Position = MVP * vec4(worldPos, 1.0);
            vColor = iColor;
        }
        """
        fragment_src = """
        #version 330 core
        in vec3 vColor;
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(vColor, 1.0);
        }
        """
        vertex = compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        return compileProgram(vertex, fragment)

    # --------------------
    # Cube geometry
    # --------------------
    def create_cube_vao(self):
        s = self.voxel_size / 2
        vertices = np.array([
            -s, -s,  s,
            s, -s,  s,
            s,  s,  s,
            -s,  s,  s,
            -s, -s, -s,
            s, -s, -s,
            s,  s, -s,
            -s,  s, -s,
        ], dtype=np.float32)

        indices = np.array([
            0,1,2, 2,3,0,
            4,6,5, 4,7,6,
            4,0,3, 3,7,4,
            1,5,6, 6,2,1,
            3,2,6, 6,7,3,
            4,5,1, 1,0,4
        ], dtype=np.uint32)

        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(0)
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindVertexArray(0)
        return VAO, len(indices)

    # --------------------
    # Instance data
    # --------------------
    def get_surface_instances(self):
        positions, colors = [], []
        snapshot = self.snapshots[0]['spins']
        for i,j,k in self.surface_indices:
            x = (i - self.H/2) * self.voxel_size + self.voxel_size/2
            y = (j - self.W/2) * self.voxel_size + self.voxel_size/2
            z = (k - self.L/2) * self.voxel_size + self.voxel_size/2
            positions.append([x,y,z])
            colors.append(COLOR_SPIN_UP if snapshot[i,j,k]==1 else COLOR_SPIN_DOWN)
        return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.float32)

    def update_instance_buffers(self):
        glBindVertexArray(self.VAO)
        self.instanceVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instanceVBO)
        glBufferData(GL_ARRAY_BUFFER, self.instance_positions.nbytes, self.instance_positions, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,12,None)
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1,1)

        self.colorVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorVBO)
        glBufferData(GL_ARRAY_BUFFER, self.instance_colors.nbytes, self.instance_colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,12,None)
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2,1)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        glBindVertexArray(0)

    def update_colors(self):
        snapshot = self.snapshots[self.current_idx]['spins']
        for idx,(i,j,k) in enumerate(self.surface_indices):
            self.instance_colors[idx] = COLOR_SPIN_UP if snapshot[i,j,k]==1 else COLOR_SPIN_DOWN
        glBindBuffer(GL_ARRAY_BUFFER, self.colorVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.instance_colors.nbytes, self.instance_colors)
        glBindBuffer(GL_ARRAY_BUFFER,0)

    # --------------------
    # Rendering
    # --------------------
    def perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(fov/2)
        return np.array([
            [f/aspect,0,0,0],
            [0,f,0,0],
            [0,0,(far+near)/(near-far),(2*far*near)/(near-far)],
            [0,0,-1,0]
        ], dtype=np.float32)

    def translate(self,x,y,z):
        return np.array([
            [1,0,0,x],
            [0,1,0,y],
            [0,0,1,z],
            [0,0,0,1]
        ], dtype=np.float32)

    def rotate_y(self,angle):
        c,s = math.cos(angle), math.sin(angle)
        return np.array([
            [c,0,s,0],
            [0,1,0,0],
            [-s,0,c,0],
            [0,0,0,1]
        ], dtype=np.float32)

    def rotate_x(self,angle):
        c,s = math.cos(angle), math.sin(angle)
        return np.array([
            [1,0,0,0],
            [0,c,-s,0],
            [0,s,c,0],
            [0,0,0,1]
        ], dtype=np.float32)

    # ----------------------------
    # Text drawing utility
    # ----------------------------
    def draw_text(self, text, x, y, cache=False):
        """Draw text at (x,y). If cache=True, store surface for reuse."""
        if cache and text in self.text_surfaces:
            surf, data, w, h = self.text_surfaces[text]
        else:
            surf = self.font.render(text, True, (255, 255, 255))
            data = pygame.image.tostring(surf, "RGBA", True)
            w, h = surf.get_width(), surf.get_height()
            if cache:
                self.text_surfaces[text] = (surf, data, w, h)
        glWindowPos2d(x, y)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)


    # ----------------------------
    # Label + dynamic value helper
    # ----------------------------
    def draw_text_label_value(self, key, x, y, value=None, fmt="{:.4f}"):
        """Draw a cached label and a dynamic value next to it."""
        self.draw_text(f"{key}: ", x, y, cache=True)
        if value is not None:
            value_str = fmt.format(value) if isinstance(value, float) else str(value)
            self.draw_text(value_str, x + self.text_surfaces[f"{key}: "][2], y)


    # ----------------------------
    # Render dynamic simulation data
    # ----------------------------
    def render_timestep(self):
        snapshot = self.snapshots[self.current_idx]
        x, y = 10, 10
        spacing = 18
        self.draw_text_label_value("Step", x, y, snapshot['step'], fmt="{}")
        y += spacing
        self.draw_text_label_value("Energy", x, y, snapshot['energy'])
        y += spacing
        self.draw_text_label_value("Magnetization", x, y, snapshot['magnetization'])
        y += spacing
        self.draw_text_label_value("Temperature", x, y, snapshot['temperature'])


    # ----------------------------
    # Render metadata
    # ----------------------------
    def render_metadata(self):
        lines = [
            f"Metadata:",
            f"Lattice: {self.metadata['H']}x{self.metadata['W']}x{self.metadata['L']}",
            f"Start T: {self.metadata['start_T']:.2f}",
            f"Target T: {self.metadata['target_T']:.2f}",
            f"dT: {self.metadata['dT']:.4f} every {self.metadata['steps_dT']} steps",
            f"Coupling J: {self.metadata['J']:.2f}",
            f"External Field B: {self.metadata['B']:.2f}",
            f"Total Steps: {self.metadata['steps']}",
            f"Save Every: {self.metadata['save_every']}",
        ]
        y_offset = 15
        for line in lines:
            self.draw_text(line, 0, WINDOW_HEIGHT - y_offset)
            y_offset += 16


    # ----------------------------
    # Render magnetization bar
    # ----------------------------
    def render_magnetization_bar(self):
        mag = self.snapshots[self.current_idx]["magnetization"]
        max_width = 100
        bar_height = 10
        bar_width = int(mag * max_width)

        x_center = WINDOW_WIDTH // 2
        y_base = WINDOW_HEIGHT - 40

        # static labels
        self.draw_text("Magnetization", x_center - 30, y_base + 18)
        self.draw_text("-1", x_center - max_width - 10, y_base - 5)
        self.draw_text("+1", x_center + max_width - 5, y_base - 5)

        if bar_width == 0:
            return

        # dynamic bar
        color = COLOR_SPIN_UP if bar_width > 0 else COLOR_SPIN_DOWN
        bar_array = np.zeros((bar_height, abs(bar_width), 3), dtype=np.uint8)
        bar_array[:] = (np.array(color) * 255).astype(np.uint8)

        # No need to flip the array
        if bar_width > 0:
            glWindowPos2d(x_center, y_base)
        else:
            glWindowPos2d(x_center + bar_width, y_base)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glDrawPixels(abs(bar_width), bar_height, GL_RGB, GL_UNSIGNED_BYTE, bar_array)

    def update_frame(self):
        if not self.playing:
            return

        now = time.time()
        if now - self.last_frame_time >= 1.0 / self.playback_fps:
            if self.reverse:
                self.current_idx = (self.current_idx - 1) % self.num_snapshots
            else:
                self.current_idx = (self.current_idx + 1) % self.num_snapshots
            self.update_colors()
            self.last_frame_time = now

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

        # camera + projection
        aspect = WINDOW_WIDTH / WINDOW_HEIGHT
        proj = self.perspective(math.radians(100), aspect, 0.1, 1000.0)
        camera_dist = 2.5 * max(self.H, self.W, self.L) * self.voxel_size
        view = self.translate(0,0,-camera_dist)
        angle = time.time() * 0.4
        model = self.rotate_y(angle) @ self.rotate_x(angle * 0.2)
        MVP = proj @ view @ model
        loc = glGetUniformLocation(self.shader_program,"MVP")
        glUniformMatrix4fv(loc,1,GL_TRUE,MVP)

        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(GL_TRIANGLES,self.num_indices,GL_UNSIGNED_INT,None,len(self.instance_positions))
        glBindVertexArray(0)
        glUseProgram(0)

        # overlays
        self.render_timestep()
        self.render_metadata()
        self.render_magnetization_bar()

        pygame.display.flip()



def load_snapshots(open_dir):
    data = np.load(os.path.join(open_dir, "snapshots.npz"), allow_pickle=True)

    # --- handle multi-temperature storage ---
    if "all_snapshots" in data.files:
        all_snapshots = data["all_snapshots"].item()  # dict {T: [snapshots]}
        # pick the first temperature by default
        temp = sorted(all_snapshots.keys())[0]
        snapshots = all_snapshots[temp]
    else:
        snapshots = data["snapshots"]

    # --- safe metadata extraction ---
    metadata = {}
    for key in data.files:
        if key in ("snapshots", "all_snapshots"):
            continue
        arr = data[key]
        if arr.shape == ():        # scalar
            metadata[key] = arr.item()
        else:                      # array
            metadata[key] = arr

    return snapshots, metadata


def run_visualizer(open_dir):
    open_dir = resolve_open_dir(open_dir)
    snapshots, metadata = load_snapshots(open_dir)
    viz = IsingVisualizer3D(snapshots, metadata)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # pause/resume
                    viz.playing = not viz.playing
                elif event.key == pygame.K_r:  # reverse
                    viz.reverse = not viz.reverse
                elif event.key == pygame.K_UP:  # faster
                    viz.playback_fps *= 1.5
                elif event.key == pygame.K_DOWN:  # slower
                    viz.playback_fps /= 1.5

        viz.update_frame()
        viz.render()
        clock.tick(60)

    pygame.quit()

# ================================
# MAIN LOOP
# ================================
def main():
    parser = argparse.ArgumentParser(description="3D Ising Model Simulation")
    parser.add_argument("--open", type=str, default="ising_sim", help="Directory")
    args = parser.parse_args()
    run_visualizer(args.open)

if __name__=="__main__":
    main()
