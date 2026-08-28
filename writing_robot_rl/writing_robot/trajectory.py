"""
Trajectory utilities: SVG parsing, interpolation, path planning.
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path

try:
    from svgpathtools import svg2paths, Path as SVGPath, Line, CubicBezier, QuadraticBezier, Arc
    SVG_AVAILABLE = True
except ImportError:
    SVG_AVAILABLE = False
    print("Warning: svgpathtools not installed. SVG parsing unavailable.")


def interpolate_trajectory(
    waypoints: np.ndarray, 
    density: float = 1.0,
    closed: bool = False
) -> np.ndarray:
    """
    Interpolate sparse waypoints to dense path with given density (mm between points).
    
    Args:
        waypoints: (N, 2) array of corner points
        density: target distance between interpolated points (mm)
        closed: whether path is closed (return to start)
    
    Returns:
        (M, 2) dense trajectory
    """
    if len(waypoints) < 2:
        return waypoints
    
    segments = []
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < density:
            segments.append(p0.reshape(1, 2))
        else:
            n_points = max(2, int(np.ceil(seg_len / density)))
            t = np.linspace(0, 1, n_points)
            seg_pts = p0 + t[:, None] * seg_vec
            segments.append(seg_pts)
    
    if closed and len(waypoints) > 2:
        p0 = waypoints[-1]
        p1 = waypoints[0]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len >= density:
            n_points = max(2, int(np.ceil(seg_len / density)))
            t = np.linspace(0, 1, n_points)
            seg_pts = p0 + t[:, None] * seg_vec
            segments.append(seg_pts)
    
    if not segments:
        return waypoints
    
    # Combine, avoiding duplicate points at joints
    result = [segments[0]]
    for seg in segments[1:]:
        result.append(seg[1:])  # skip first point (duplicate of previous end)
    
    return np.vstack(result).astype(np.float32)


def svg_to_trajectory(
    svg_path: Union[str, Path],
    density: float = 1.0,
    scale: float = 1.0,
    translate: Tuple[float, float] = (0.0, 0.0),
    flip_y: bool = True,
) -> np.ndarray:
    """
    Convert SVG file to dense trajectory waypoints.
    
    Args:
        svg_path: Path to SVG file
        density: mm between interpolated points
        scale: Scale factor (SVG units -> mm)
        translate: (dx, dy) offset in mm
        flip_y: Flip Y axis (SVG origin top-left -> robot origin bottom-left)
    
    Returns:
        (N, 2) trajectory array in mm
    """
    if not SVG_AVAILABLE:
        raise RuntimeError("svgpathtools not installed: pip install svgpathtools")
    
    paths, attributes = svg2paths(str(svg_path))
    
    all_points = []
    
    for path, attr in zip(paths, attributes):
        # Sample path at regular intervals
        path_length = path.length()
        if path_length == 0:
            continue
        
        n_samples = max(2, int(np.ceil(path_length * scale / density)))
        t_values = np.linspace(0, 1, n_samples)
        
        for t in t_values:
            point = path.point(t)
            x = point.real * scale
            y = point.imag * scale
            
            if flip_y:
                y = -y
            
            x += translate[0]
            y += translate[1]
            
            all_points.append([x, y])
    
    if not all_points:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    
    trajectory = np.array(all_points, dtype=np.float32)
    
    # Re-interpolate to ensure uniform density
    return interpolate_trajectory(trajectory, density=density)


def trajectory_from_text(
    text: str,
    font_size: float = 20.0,
    density: float = 1.0,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate trajectory from text using a TTF font.
    Requires freetype-py: pip install freetype-py
    """
    try:
        import freetype
    except ImportError:
        raise RuntimeError("freetype-py not installed: pip install freetype-py")
    
    if font_path is None:
        # Try to find a default font
        import os
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for f in font_candidates:
            if os.path.exists(f):
                font_path = f
                break
        if font_path is None:
            raise RuntimeError("No TTF font found. Provide font_path.")
    
    face = freetype.Face(font_path)
    face.set_char_size(int(font_size * 64))
    
    all_contours = []
    x_offset = 0.0
    
    for char in text:
        face.load_char(char)
        glyph = face.glyph
        outline = glyph.outline
        
        # Extract contours from outline
        contours = []
        points = []
        for i in range(outline.n_points):
            pt = outline.points[i]
            points.append([pt.x / 64.0, pt.y / 64.0])
        
        # Convert to contours based on end_points
        start = 0
        for end in outline.end_points:
            contour_pts = points[start:end+1]
            if len(contour_pts) > 1:
                # Scale and offset
                contour_pts = np.array(contour_pts) * (font_size / face.units_per_EM)
                contour_pts[:, 0] += x_offset
                all_contours.append(contour_pts)
            start = end + 1
        
        # Advance x_offset by advance width
        x_offset += glyph.advance.x / 64.0
    
    if not all_contours:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    
    # Combine all contours with pen-up moves (NaN separators)
    combined = []
    for i, contour in enumerate(all_contours):
        if i > 0:
            combined.append([np.nan, np.nan])  # pen up separator
        combined.append(contour)
    
    trajectory = np.vstack(combined)
    trajectory = trajectory[~np.isnan(trajectory).any(axis=1)]  # remove separators for interpolation
    
    return interpolate_trajectory(trajectory, density=density, closed=False)


def smooth_trajectory(
    trajectory: np.ndarray,
    window_size: int = 5,
    method: str = "moving_average"
) -> np.ndarray:
    """
    Smooth trajectory to reduce jerk.
    
    Args:
        trajectory: (N, 2) array
        window_size: Smoothing window (odd number)
        method: "moving_average", "gaussian", "savgol"
    
    Returns:
        Smoothed trajectory (N, 2)
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    
    if len(trajectory) < window_size:
        return trajectory
    
    if method == "moving_average":
        kernel = np.ones(window_size) / window_size
        x_smooth = np.convolve(trajectory[:, 0], kernel, mode='same')
        y_smooth = np.convolve(trajectory[:, 1], kernel, mode='same')
    elif method == "gaussian":
        sigma = window_size / 3.0
        x_smooth = gaussian_filter1d(trajectory[:, 0], sigma=sigma)
        y_smooth = gaussian_filter1d(trajectory[:, 1], sigma=sigma)
    elif method == "savgol":
        poly_order = min(3, window_size - 1)
        x_smooth = savgol_filter(trajectory[:, 0], window_size, poly_order)
        y_smooth = savgol_filter(trajectory[:, 1], window_size, poly_order)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.column_stack([x_smooth, y_smooth]).astype(np.float32)


def resample_trajectory(
    trajectory: np.ndarray,
    num_points: int,
    method: str = "linear"
) -> np.ndarray:
    """
    Resample trajectory to fixed number of points.
    """
    from scipy.interpolate import interp1d
    
    if len(trajectory) < 2:
        return trajectory
    
    # Cumulative arc length parameterization
    diffs = np.diff(trajectory, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    
    if total_length == 0:
        return trajectory
    
    new_params = np.linspace(0, total_length, num_points)
    
    fx = interp1d(cum_length, trajectory[:, 0], kind=method)
    fy = interp1d(cum_length, trajectory[:, 1], kind=method)
    
    return np.column_stack([fx(new_params), fy(new_params)]).astype(np.float32)


def generate_basic_shapes() -> dict:
    """Generate common test trajectories."""
    shapes = {}
    
    # Square
    shapes["square"] = np.array([
        [10, 10], [90, 10], [90, 90], [10, 90], [10, 10]
    ], dtype=np.float32)
    
    # Circle
    angles = np.linspace(0, 2*np.pi, 100)
    shapes["circle"] = np.column_stack([
        50 + 40 * np.cos(angles),
        50 + 40 * np.sin(angles)
    ]).astype(np.float32)
    
    # Figure 8
    t = np.linspace(0, 2*np.pi, 200)
    shapes["figure8"] = np.column_stack([
        50 + 35 * np.sin(t),
        50 + 25 * np.sin(2*t)
    ]).astype(np.float32)
    
    # Spiral
    t = np.linspace(0, 4*np.pi, 300)
    r = 5 + t * 5
    shapes["spiral"] = np.column_stack([
        50 + r * np.cos(t),
        50 + r * np.sin(t)
    ]).astype(np.float32)
    
    # Interpolate all
    for k, v in shapes.items():
        shapes[k] = interpolate_trajectory(v, density=1.0)
    
    return shapes