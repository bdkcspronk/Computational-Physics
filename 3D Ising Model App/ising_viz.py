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
import re
from ising_initialization import resolve_open_dir

# ================================
# Visualization parameters
# ================================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
VOXEL_SIZE = 1.0
COLOR_SPIN_UP = (0.8, 0.8, 0.8)
COLOR_SPIN_DOWN = (0.3, 0.5, 0.5)
COLOR_BACKGROUND = (0.1, 0.1, 0.1, 1.0)

spacing_factor = 2.5  # how much space to put between cubes (relative to cube size)

PLAYBACK_FPS = 50  # how fast to advance snapshots per second

# ================================
# OpenGL visualizer
# ================================
class IsingVisualizer3D:
    def __init__(self, temp_snapshots, metadata, voxel_size=VOXEL_SIZE):
        self.metadata = metadata
        self.voxel_size = voxel_size

        self.cubes = []
        for idx, (temp, snaps) in enumerate(temp_snapshots):
            snaps_list = list(snaps)
            if len(snaps_list) == 0:
                continue
            self.cubes.append({
                "temperature": float(temp),
                "snapshots": snaps_list,
                "num_snapshots": len(snaps_list),
                "current_idx": 0,
                "rotation_phase": 0.0,
                "rotation_speed": 0.4,
            })

        if not self.cubes:
            raise ValueError("No snapshots to visualize.")

        self.num_cubes = len(self.cubes)
        self.show_single_details = self.num_cubes == 1

        self.last_frame_time = time.time()
        self.playing = True
        self.reverse = False
        self.playback_fps = PLAYBACK_FPS  # can change with buttons

        first_spins = self.cubes[0]["snapshots"][0]["spins"]
        self.H, self.W, self.L = first_spins.shape

        # Precompute surface indices
        self.surface_indices = [
            (i, j, k) for i in range(self.H)
                      for j in range(self.W)
                      for k in range(self.L)
                      if i == 0 or i == self.H - 1 or j == 0 or j == self.W - 1 or k == 0 or k == self.L - 1
        ]
        self.surface_coords = np.array(self.surface_indices, dtype=np.int32)
        self.surface_flat_idx = np.ravel_multi_index(
            self.surface_coords.T, (self.H, self.W, self.L)
        )
        self.color_lut = np.array([COLOR_SPIN_DOWN, COLOR_SPIN_UP], dtype=np.float32)

        # Pygame + OpenGL init
        pygame.init()
        pygame.font.init()

        self.font = pygame.font.SysFont("Arial", 12)  # name, size
        self.text_surfaces = {}
        self.text_height = self.font.get_linesize()

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
        self.mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        self.VAO, self.num_indices = self.create_cube_vao()
        self.instance_positions = self.get_surface_positions()
        self.instance_colors = self.get_colors_for_snapshot(self.cubes[0]["snapshots"][0])
        self.update_instance_buffers()

        self.metadata_lines = self.build_metadata_lines()
        self.metadata_height = 16 * len(self.metadata_lines) + 10

        (
            self.grid_positions,
            grid_w,
            grid_h,
            self.cube_extent,
            self.spacing,
        ) = self.compute_grid_layout(self.num_cubes)

        self.grid_cols = max(1, math.ceil(math.sqrt(self.num_cubes)))
        self.grid_rows = max(1, math.ceil(self.num_cubes / self.grid_cols))
        usable_height = max(1.0, WINDOW_HEIGHT - self.metadata_height - 10)
        self.cell_width = WINDOW_WIDTH / self.grid_cols
        self.cell_height = usable_height / self.grid_rows
        self.screen_centers = [(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2) for _ in range(self.num_cubes)]
        self.overlay_positions = [{"temp_y": WINDOW_HEIGHT / 2, "bar_y": WINDOW_HEIGHT / 2} for _ in range(self.num_cubes)]

        self.proj = self.perspective(math.radians(45), WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0)
        base_dist = 2.2 * max(grid_w, grid_h)
        if self.num_cubes == 1:
            self.camera_dist = base_dist * 1.6
        else:
            self.camera_dist = base_dist
        self.view = self.translate(0, 0, -self.camera_dist)

        self.max_bar_width = 24
        self.bar_height = 4
        self.bar_buffer = np.zeros((self.bar_height, self.max_bar_width, 3), dtype=np.uint8)
        self.overlay_offset = max(20, int(self.cell_height * 0.32))
        self.label_offset = max(22, int(self.cell_height * 0.34))
        self.temp_padding = max(6, int(self.text_height * 0.4))
        self.bar_padding = max(6, int(self.text_height * 0.4))
        self.max_bar_width = self.compute_bar_width_limit(self.max_bar_width)

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
            0, 1, 2, 2, 3, 0,
            4, 6, 5, 4, 7, 6,
            4, 0, 3, 3, 7, 4,
            1, 5, 6, 6, 2, 1,
            3, 2, 6, 6, 7, 3,
            4, 5, 1, 1, 0, 4
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
    # Layout helpers
    # --------------------
    def build_metadata_lines(self):
        def fmt(value, fmt_str="{:.2f}"):
            try:
                return fmt_str.format(float(value))
            except (TypeError, ValueError):
                return str(value)

        relax_perc = self.metadata.get("relax_perc", "?")
        if isinstance(relax_perc, (int, float)):
            relax_perc_str = f"{relax_perc:.1f}%"
        else:
            relax_perc_str = str(relax_perc)

        return [
            "Metadata:",
            f"Lattice: {self.metadata.get('H', '?')}x{self.metadata.get('W', '?')}x{self.metadata.get('L', '?')}",
            f"Start T: {fmt(self.metadata.get('start_T', '?'))}",
            f"Target T: {fmt(self.metadata.get('target_T', '?'))}",
            f"dT: {fmt(self.metadata.get('dT', '?'), '{:.4f}')} every {self.metadata.get('steps_dT', '?')} steps",
            f"Coupling J: {fmt(self.metadata.get('J', '?'))}",
            f"External Field B: {fmt(self.metadata.get('B', '?'))}",
            f"Total Steps: {self.metadata.get('steps', '?')}",
            f"Relax Perc: {relax_perc_str}",
            f"Save Every: {self.metadata.get('save_every', '?')}",
        ]

    def compute_bar_width_limit(self, fallback_width):
        half_extent = self.cube_extent * 0.5 * math.sqrt(2)
        if self.camera_dist <= 0:
            return fallback_width
        f = 1.0 / math.tan(math.radians(45) / 2.0)
        aspect = WINDOW_WIDTH / WINDOW_HEIGHT
        ndc_half_width = (f / aspect) * (half_extent / self.camera_dist)
        screen_half_width = ndc_half_width * (WINDOW_WIDTH / 2.0)
        max_half_width = max(10.0, screen_half_width * 0.48)
        return min(fallback_width, max_half_width)

    def compute_grid_layout(self, n):
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))

        cube_extent = max(self.H, self.W, self.L) * self.voxel_size
        spacing = cube_extent * spacing_factor

        x0 = - (cols - 1) * spacing / 2
        y0 = (rows - 1) * spacing / 2

        positions = []
        for idx in range(n):
            row = idx // cols
            col = idx % cols
            x = x0 + col * spacing
            y = y0 - row * spacing
            positions.append((x, y, 0.0))

        grid_w = (cols - 1) * spacing + cube_extent
        grid_h = (rows - 1) * spacing + cube_extent

        return positions, grid_w, grid_h, cube_extent, spacing

    # --------------------
    # Instance data
    # --------------------
    def get_surface_positions(self):
        coords = self.surface_coords.astype(np.float32)
        positions = np.empty_like(coords, dtype=np.float32)
        positions[:, 0] = (coords[:, 0] - self.H / 2) * self.voxel_size + self.voxel_size / 2
        positions[:, 1] = (coords[:, 1] - self.W / 2) * self.voxel_size + self.voxel_size / 2
        positions[:, 2] = (coords[:, 2] - self.L / 2) * self.voxel_size + self.voxel_size / 2
        return positions

    def get_colors_for_snapshot(self, snapshot):
        spins_flat = snapshot["spins"].ravel()[self.surface_flat_idx]
        mask = (spins_flat > 0).astype(np.int32)
        return self.color_lut[mask].astype(np.float32)

    def update_instance_buffers(self):
        glBindVertexArray(self.VAO)
        self.instanceVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instanceVBO)
        glBufferData(GL_ARRAY_BUFFER, self.instance_positions.nbytes, self.instance_positions, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)

        self.colorVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorVBO)
        glBufferData(GL_ARRAY_BUFFER, self.instance_colors.nbytes, self.instance_colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def update_colors(self, snapshot):
        self.instance_colors[:] = self.get_colors_for_snapshot(snapshot)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.instance_colors.nbytes, self.instance_colors)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # --------------------
    # Rendering
    # --------------------
    def perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(fov / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def translate(self, x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def rotate_y(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def rotate_x(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    # ----------------------------
    # Text drawing utility
    # ----------------------------
    def get_text_surface(self, text, cache=False):
        if cache and text in self.text_surfaces:
            return self.text_surfaces[text]
        surf = self.font.render(text, True, (255, 255, 255))
        data = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_width(), surf.get_height()
        if cache:
            self.text_surfaces[text] = (surf, data, w, h)
        return surf, data, w, h

    def draw_text(self, text, x, y, cache=False):
        _, data, w, h = self.get_text_surface(text, cache=cache)
        glWindowPos2d(int(x), int(y))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw_text_centered(self, text, x_center, y, cache=False):
        _, data, w, h = self.get_text_surface(text, cache=cache)
        glWindowPos2d(int(x_center - w / 2), int(y))
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
    def render_timestep(self, snapshot):
        x, y = 10, 10
        spacing = 18
        self.draw_text_label_value("Step", x, y, snapshot["step"], fmt="{}")
        y += spacing
        self.draw_text_label_value("Energy", x, y, snapshot["energy"])
        y += spacing
        self.draw_text_label_value("Magnetization", x, y, snapshot["magnetization"])
        y += spacing
        self.draw_text_label_value("Temperature", x, y, snapshot["temperature"])

    # ----------------------------
    # Render metadata
    # ----------------------------
    def render_metadata(self):
        y_offset = 15
        for line in self.metadata_lines:
            self.draw_text(line, 0, WINDOW_HEIGHT - y_offset)
            y_offset += 16

    # ----------------------------
    # Render magnetization bar
    # ----------------------------
    def render_magnetization_bar_at(self, mag, x_center, y_base):
        if self.max_bar_width <= 0 or self.bar_height <= 0:
            return

        try:
            mag_val = float(mag)
        except (TypeError, ValueError):
            return

        mag_val = max(-1.0, min(1.0, mag_val))
        bar_width = int(mag_val * self.max_bar_width)

        number_y = y_base - 3
        tick_y = y_base - 3
        self.draw_text("-", x_center - self.max_bar_width - 10, number_y, cache=True)
        self.draw_text("+", x_center + self.max_bar_width + 8, number_y, cache=True)
        self.draw_text("|", x_center, tick_y, cache=True)
        self.draw_text("|", x_center - self.max_bar_width, tick_y, cache=True)
        self.draw_text("|", x_center + self.max_bar_width, tick_y, cache=True)

        if bar_width == 0:
            return

        color = COLOR_SPIN_UP if bar_width > 0 else COLOR_SPIN_DOWN
        width = abs(bar_width)
        self.bar_buffer[:, :width, :] = (np.array(color) * 255).astype(np.uint8)

        if bar_width > 0:
            glWindowPos2d(int(x_center), int(y_base))
        else:
            glWindowPos2d(int(x_center + bar_width), int(y_base))

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glDrawPixels(width, self.bar_height, GL_RGB, GL_UNSIGNED_BYTE, self.bar_buffer[:, :width, :])

    def render_cube_overlays(self):
        for idx, cube in enumerate(self.cubes):
            if cube["num_snapshots"] == 0:
                continue

            snapshot = cube["snapshots"][cube["current_idx"]]
            mag = snapshot["magnetization"]
            center_x, center_y = self.screen_centers[idx]

            label = f"T={cube['temperature']:.2f}"
            label_y = self.overlay_positions[idx]["temp_y"]
            self.draw_text_centered(label, center_x, label_y, cache=True)

            bar_y = self.overlay_positions[idx]["bar_y"]
            self.render_magnetization_bar_at(mag, center_x, bar_y)

    def render_step_counter(self):
        if not self.cubes:
            return
        cube = self.cubes[0]
        if cube["num_snapshots"] == 0:
            return
        step_idx = cube["current_idx"] + 1
        total = cube["num_snapshots"]
        text = f"Step: {step_idx}/{total}"
        self.draw_text(text, 10, WINDOW_HEIGHT - self.metadata_height - 20, cache=False)

    def update_frame(self):
        if not self.playing:
            return

        now = time.time()
        if now - self.last_frame_time >= 1.0 / self.playback_fps:
            for cube in self.cubes:
                if cube["num_snapshots"] == 0:
                    continue
                if self.reverse:
                    cube["current_idx"] = (cube["current_idx"] - 1) % cube["num_snapshots"]
                else:
                    cube["current_idx"] = (cube["current_idx"] + 1) % cube["num_snapshots"]
            self.last_frame_time = now

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

        angle_base = time.time()
        glBindVertexArray(self.VAO)

        for idx, cube in enumerate(self.cubes):
            snapshot = cube["snapshots"][cube["current_idx"]]
            self.update_colors(snapshot)

            angle = angle_base * cube["rotation_speed"] + cube["rotation_phase"]
            model = self.translate(*self.grid_positions[idx]) @ self.rotate_y(angle) @ self.rotate_x(angle * 0.2)

            local_center = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            world_center = model @ local_center
            view_center = self.view @ world_center
            clip_center = self.proj @ view_center
            if clip_center[3] != 0:
                ndc = clip_center[:3] / clip_center[3]
                screen_x = (ndc[0] + 1.0) / 2.0 * WINDOW_WIDTH
                screen_y = (ndc[1] + 1.0) / 2.0 * WINDOW_HEIGHT
                self.screen_centers[idx] = (screen_x, screen_y)

            half = self.cube_extent * 0.5
            corners = [
                (-half, -half, -half, 1.0), (-half, -half, half, 1.0),
                (-half, half, -half, 1.0), (-half, half, half, 1.0),
                (half, -half, -half, 1.0), (half, -half, half, 1.0),
                (half, half, -half, 1.0), (half, half, half, 1.0),
            ]
            min_y = float("inf")
            max_y = float("-inf")
            for cx, cy, cz, cw in corners:
                clip = self.proj @ (self.view @ (model @ np.array([cx, cy, cz, cw], dtype=np.float32)))
                if clip[3] == 0:
                    continue
                ndc_corner = clip[:3] / clip[3]
                screen_y_corner = (ndc_corner[1] + 1.0) / 2.0 * WINDOW_HEIGHT
                min_y = min(min_y, screen_y_corner)
                max_y = max(max_y, screen_y_corner)
            if min_y != float("inf") and max_y != float("-inf"):
                self.overlay_positions[idx]["temp_y"] = max_y + self.temp_padding
                self.overlay_positions[idx]["bar_y"] = min_y - self.bar_padding - self.bar_height

            MVP = self.proj @ self.view @ model
            glUniformMatrix4fv(self.mvp_loc, 1, GL_TRUE, MVP)

            glDrawElementsInstanced(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None, len(self.instance_positions))

        glBindVertexArray(0)
        glUseProgram(0)

        # overlays
        self.render_cube_overlays()
        if self.show_single_details:
            self.render_timestep(self.cubes[0]["snapshots"][self.cubes[0]["current_idx"]])
        self.render_step_counter()
        self.render_metadata()

        pygame.display.flip()

def load_snapshots(open_path):
    if os.path.isdir(open_path):
        file_path = os.path.join(open_path, "snapshots.npz")
    else:
        file_path = open_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(resolve_open_dir(), file_path)

    data = np.load(file_path, allow_pickle=True)

    temp_entries = []

    def extract_temp(key):
        try:
            return float(key)
        except (TypeError, ValueError):
            pass
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(key))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None

    # --- handle multi-temperature storage ---
    if "all_snapshots" in data.files:
        all_snapshots = data["all_snapshots"].item()  # dict {T: [snapshots]}
        for key, snaps in all_snapshots.items():
            temp = extract_temp(key)
            if temp is None:
                continue
            temp_entries.append((temp, len(temp_entries), snaps))
    else:
        for key in data.files:
            if key in ("snapshots", "all_snapshots"):
                continue
            temp = extract_temp(key)
            if temp is None:
                continue
            temp_entries.append((temp, len(temp_entries), data[key]))

        if not temp_entries and "snapshots" in data.files:
            temp = data["start_T"].item() if "start_T" in data.files else 0.0
            temp_entries.append((float(temp), len(temp_entries), data["snapshots"]))

    temp_entries.sort(key=lambda item: (item[0], item[1]))
    temp_snapshots = [(temp, list(snaps)) for temp, _, snaps in temp_entries]

    # --- safe metadata extraction ---
    metadata = {}
    for key in data.files:
        if key in ("snapshots", "all_snapshots"):
            continue
        try:
            float(key)
            continue
        except ValueError:
            pass
        arr = data[key]
        if arr.shape == ():        # scalar
            metadata[key] = arr.item()
        else:                      # array
            metadata[key] = arr

    return temp_snapshots, metadata


def run_visualizer(open_path):
    if os.path.isdir(open_path):
        open_path = resolve_open_dir(open_path)
    temp_snapshots, metadata = load_snapshots(open_path)
    if not temp_snapshots:
        print("No snapshots found to visualize.")
        return

    viz = IsingVisualizer3D(temp_snapshots, metadata)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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

if __name__ == "__main__":
    main()
