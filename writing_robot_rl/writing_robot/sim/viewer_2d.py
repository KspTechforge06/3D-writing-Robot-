"""
2D Pygame Viewer for Writing Robot Simulation

Real-time visualization of robot position, trajectory, and pen state.
"""
import pygame
import numpy as np
from typing import Optional, Tuple
import math


class Viewer2D:
    """
    Pygame-based 2D viewer for the writing robot.
    
    Features:
    - Robot position (with pen up/down visualization)
    - Target trajectory
    - Path progress indicator
    - Current waypoint highlight
    - Reward/episode info overlay
    """
    
    def __init__(
        self,
        window_size: Tuple[int, int] = (800, 600),
        x_range: Tuple[float, float] = (0.0, 100.0),
        y_range: Tuple[float, float] = (0.0, 100.0),
        target_path: Optional[np.ndarray] = None,
        title: str = "Writing Robot Simulator",
        background_color: Tuple[int, int, int] = (25, 25, 35),
        trajectory_color: Tuple[int, int, int] = (80, 180, 255),
        robot_color: Tuple[int, int, int] = (255, 100, 100),
        pen_down_color: Tuple[int, int, int] = (255, 50, 50),
        pen_up_color: Tuple[int, int, int] = (100, 200, 100),
        waypoint_color: Tuple[int, int, int] = (255, 255, 0),
        grid_color: Tuple[int, int, int] = (50, 50, 70),
        text_color: Tuple[int, int, int] = (220, 220, 220),
    ):
        self.window_size = window_size
        self.x_range = x_range
        self.y_range = y_range
        self.target_path = target_path
        
        # Colors
        self.bg_color = background_color
        self.traj_color = trajectory_color
        self.robot_color = robot_color
        self.pen_down_color = pen_down_color
        self.pen_up_color = pen_up_color
        self.waypoint_color = waypoint_color
        self.grid_color = grid_color
        self.text_color = text_color
        
        # State
        self.position = np.array([x_range[0], y_range[0]], dtype=np.float32)
        self.pen_state = 1.0
        self.path_progress = 0.0
        self.current_waypoint_idx = 0
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.reward_components = {}
        
        # Pygame setup
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_small = pygame.font.SysFont("monospace", 11)
        self.font_big = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Coordinate transform
        self._update_transform()
        
        # Trail
        self.trail = []
        self.max_trail = 500
    
    def _update_transform(self):
        """Compute world-to-screen transform."""
        w, h = self.window_size
        margin = 40
        self.scale_x = (w - 2 * margin) / (self.x_range[1] - self.x_range[0])
        self.scale_y = (h - 2 * margin) / (self.y_range[1] - self.y_range[0])
        self.scale = min(self.scale_x, self.scale_y)
        self.offset_x = margin + (w - 2 * margin - self.scale * (self.x_range[1] - self.x_range[0])) / 2
        self.offset_y = margin + (h - 2 * margin - self.scale * (self.y_range[1] - self.y_range[0])) / 2
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (mm) to screen pixels."""
        sx = int(self.offset_x + (x - self.x_range[0]) * self.scale)
        sy = int(self.offset_y + (self.y_range[1] - y) * self.scale)  # flip Y
        return sx, sy
    
    def update(
        self,
        position: np.ndarray,
        target_path: Optional[np.ndarray] = None,
        pen_state: float = 1.0,
        path_progress: float = 0.0,
        current_waypoint: int = 0,
        episode: int = 0,
        step: int = 0,
        total_reward: float = 0.0,
        reward_components: Optional[dict] = None,
    ):
        """Update viewer state."""
        self.position = position
        if target_path is not None:
            self.target_path = target_path
        self.pen_state = pen_state
        self.path_progress = path_progress
        self.current_waypoint_idx = current_waypoint
        self.episode = episode
        self.step = step
        self.total_reward = total_reward
        if reward_components:
            self.reward_components = reward_components
        
        # Add to trail
        self.trail.append(position.copy())
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
    
    def render(
        self,
        position: np.ndarray,
        target_path: Optional[np.ndarray] = None,
        pen_state: float = 1.0,
        path_progress: float = 0.0,
        **kwargs,
    ):
        """Render single frame."""
        self.update(position, target_path, pen_state, path_progress, **kwargs)
        self._draw()
        pygame.display.flip()
        self.clock.tick(60)
    
    def _draw(self):
        self.screen.fill(self.bg_color)
        
        # Grid
        self._draw_grid()
        
        # Trajectory
        self._draw_trajectory()
        
        # Trail (drawn path)
        self._draw_trail()
        
        # Current waypoint marker
        self._draw_current_waypoint()
        
        # Robot
        self._draw_robot()
        
        # UI overlay
        self._draw_overlay()
    
    def _draw_grid(self):
        w, h = self.window_size
        margin = 40
        grid_step_mm = 10.0
        
        # Vertical lines
        x = self.x_range[0]
        while x <= self.x_range[1]:
            sx, _ = self.world_to_screen(x, 0)
            pygame.draw.line(self.screen, self.grid_color, (sx, margin), (sx, h - margin), 1)
            x += grid_step_mm
        
        # Horizontal lines
        y = self.y_range[0]
        while y <= self.y_range[1]:
            _, sy = self.world_to_screen(0, y)
            pygame.draw.line(self.screen, self.grid_color, (margin, sy), (w - margin, sy), 1)
            y += grid_step_mm
        
        # Border
        pygame.draw.rect(self.screen, (100, 100, 120), 
                        (margin, margin, w - 2*margin, h - 2*margin), 2)
    
    def _draw_trajectory(self):
        if self.target_path is None or len(self.target_path) < 2:
            return
        
        # Draw full trajectory
        points = [self.world_to_screen(p[0], p[1]) for p in self.target_path]
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.traj_color, False, points, 2)
        
        # Draw waypoints as small dots
        for pt in self.target_path[::10]:  # every 10th
            sx, sy = self.world_to_screen(pt[0], pt[1])
            pygame.draw.circle(self.screen, self.traj_color, (sx, sy), 2)
    
    def _draw_trail(self):
        """Draw the actual path taken (pen down only)."""
        if len(self.trail) < 2:
            return
        
        # Draw segments where pen was down
        # For simplicity, draw all trail with pen-down color
        trail_points = [self.world_to_screen(p[0], p[1]) for p in self.trail]
        if len(trail_points) > 1:
            pygame.draw.lines(self.screen, self.pen_down_color, False, trail_points, 3)
    
    def _draw_current_waypoint(self):
        if self.target_path is None or len(self.target_path) == 0:
            return
        
        idx = min(self.current_waypoint_idx, len(self.target_path) - 1)
        pt = self.target_path[idx]
        sx, sy = self.world_to_screen(pt[0], pt[1])
        
        # Pulsing effect
        pulse = 8 + 4 * abs(math.sin(pygame.time.get_ticks() * 0.005))
        pygame.draw.circle(self.screen, self.waypoint_color, (sx, sy), int(pulse), 2)
        pygame.draw.circle(self.screen, self.waypoint_color, (sx, sy), 4)
    
    def _draw_robot(self):
        sx, sy = self.world_to_screen(self.position[0], self.position[1])
        
        # Robot body (square)
        robot_size = 12
        rect = pygame.Rect(sx - robot_size//2, sy - robot_size//2, robot_size, robot_size)
        
        # Color based on pen state
        color = self.pen_down_color if self.pen_state > 0.5 else self.pen_up_color
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
        
        # Direction indicator (velocity)
        vel_mag = np.linalg.norm(self.position - (self.trail[-2] if len(self.trail) > 1 else self.position))
        if vel_mag > 0.01 and len(self.trail) > 1:
            prev = self.trail[-2]
            dx = self.position[0] - prev[0]
            dy = self.position[1] - prev[1]
            angle = math.atan2(dy, dx)
            end_x = sx + int(20 * math.cos(angle))
            end_y = sy - int(20 * math.sin(angle))
            pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (end_x, end_y), 2)
            pygame.draw.circle(self.screen, (255, 255, 255), (end_x, end_y), 3)
        
        # Pen indicator
        pen_text = "PEN DOWN" if self.pen_state > 0.5 else "PEN UP"
        pen_color = self.pen_down_color if self.pen_state > 0.5 else self.pen_up_color
        text_surf = self.font_small.render(pen_text, True, pen_color)
        self.screen.blit(text_surf, (sx + 15, sy - 20))
    
    def _draw_overlay(self):
        w, h = self.window_size
        margin = 10
        
        # Episode info
        lines = [
            f"Episode: {self.episode}",
            f"Step: {self.step}",
            f"Pos: ({self.position[0]:.1f}, {self.position[1]:.1f}) mm",
            f"Progress: {self.path_progress:.1%}",
            f"Reward: {self.total_reward:.1f}",
        ]
        
        y = margin
        for line in lines:
            surf = self.font.render(line, True, self.text_color)
            self.screen.blit(surf, (margin, y))
            y += 20
        
        # Reward components
        if self.reward_components:
            y += 5
            for key, val in self.reward_components.items():
                text = f"  {key}: {val:.2f}"
                surf = self.font_small.render(text, True, (180, 180, 180))
                self.screen.blit(text_surf, (margin + 10, y))
                y += 16
        
        # Legend
        legend_y = h - 80
        pygame.draw.rect(self.screen, self.traj_color, (margin, legend_y, 20, 4))
        self.screen.blit(self.font_small.render("Target", True, self.text_color), (margin + 25, legend_y - 2))
        
        pygame.draw.rect(self.screen, self.pen_down_color, (margin, legend_y + 20, 20, 4))
        self.screen.blit(self.font_small.render("Drawn (pen down)", True, self.text_color), (margin + 25, legend_y + 18))
        
        pygame.draw.circle(self.screen, self.waypoint_color, (margin + 10, legend_y + 40), 5, 2)
        self.screen.blit(self.font_small.render("Next waypoint", True, self.text_color), (margin + 25, legend_y + 38))
    
    def get_rgb_array(self) -> np.ndarray:
        """Return current frame as RGB array for video recording."""
        # Note: pygame surface to numpy array
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
    
    def close(self):
        pygame.quit()


def create_viewer(**kwargs) -> Viewer2D:
    """Factory function."""
    return Viewer2D(**kwargs)