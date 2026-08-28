#!/usr/bin/env python3
"""Write a text string with the robot using stroke-based vector letters."""
import sys, time, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.hardware import HardwareInterface

# ---- stroke definitions (normalized 0..1 boxes) ----
LETTERS = {
    'H': [((0,0),(0,1)), ((1,0),(1,1)), ((0,0.5),(1,0.5))],
    'E': [((0,0),(0,1)), ((0,1),(1,1)), ((0,0.5),(0.7,0.5)), ((0,0),(1,0))],
    'L': [((0,0),(0,1)), ((0,0),(1,0))],
    'L': [((0,0),(0,1)), ((0,0),(1,0))],
    'O': [((0,1),(1,1)), ((1,1),(1,0)), ((1,0),(0,0)), ((0,0),(0,1))],
    'C': [((1,1),(-0.0,0.5)), ((0,0.5),(1,0))],
}

def letter_strokes(ch):
    if ch == 'O':
        # smooth rounded O via arc points
        cx, cy, r = 0.5, 0.5, 0.48
        pts = [[cx + r*np.cos(np.radians(a)), cy + r*np.sin(np.radians(a))]
               for a in np.linspace(90, -270, 80)]
        return [np.array(pts)]
    if ch == 'H':
        return [np.array([[0,0],[0,1]]), np.array([[1,0],[1,1]]),
                np.array([[0,0.5],[1,0.5]])]
    if ch == 'E':
        return [np.array([[0,0],[0,1]]), np.array([[0,1],[1,1]]),
                np.array([[0,0.5],[0.7,0.5]]), np.array([[0,0],[1,0]])]
    if ch == 'L':
        return [np.array([[0,0],[0,1]]), np.array([[0,0],[1,0]])]
    if ch == 'I':
        return [np.array([[0.5,0],[0.5,1]])]
    raise ValueError(ch)

def build_word(word, x0=3.0, y0=4.0, lw=7.5, lh=14.0, gap=2.0):
    """Return list of strokes (each Nx2 in mm, y-flipped to robot coords)."""
    out = []
    for i, ch in enumerate(word):
        cx = x0 + i*(lw + gap)
        for st in letter_strokes(ch):
            pts = np.asarray(st, dtype=float)
            pts = pts * [lw, lh] + [cx, y0]
            pts[:, 1] = y0 + lh - (pts[:, 1] - y0)  # flip y (robot origin at top-left corner)
            out.append(pts)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("word", nargs="?", default="HELLO")
    ap.add_argument("--start", type=float, default=3.0)
    ap.add_argument("--y0", type=float, default=4.0)
    ap.add_argument("--lw", type=float, default=7.5, help="letter width mm")
    ap.add_argument("--lh", type=float, default=14.0, help="letter height mm")
    run = ap.parse_args()

    strokes = build_word(run.word.upper(), run.start, run.y0, run.lw, run.lh)

    allpts = np.vstack(strokes)
    print(f"word: {run.word.upper()}  strokes: {len(strokes)}  "
          f"extent x[{allpts[:,0].min():.1f}..{allpts[:,0].max():.1f}] "
          f"y[{allpts[:,1].min():.1f}..{allpts[:,1].max():.1f}]")
    ok = allpts[:,0].min() >= 0 and allpts[:,0].max() <= 52 and \
         allpts[:,1].min() >= 0 and allpts[:,1].max() <= 49
    print("within bounds:", ok)
    if not ok:
        print("ABORTING — outside safe area")
        return

    hw = HardwareInterface(port="/dev/ttyACM0", steps_per_mm_x=5.0, steps_per_mm_y=4.8,
                           max_steps=(275, 250), safety_margin_steps=15)
    hw.connect(); hw.set_speed(40); hw.home()
    pen_up, pen_down = 90, 45
    hw.set_pen(pen_up)  # pen up before lifting to first stroke
    print("writing (with pen lift between strokes)...")
    for i, st in enumerate(strokes):
        first = st[0]
        hw.move_to_mm(first[0], first[1])  # travel to start, pen lifted
        hw.set_pen(pen_down)               # lower pen
        for x, y in st[1:]:
            hw.move_to_mm(x, y)
        hw.set_pen(pen_up)                 # lift pen
        print(f"  stroke {i+1}/{len(strokes)} done")
    hw.home()
    hw.disconnect()
    print("done ✓")

if __name__ == "__main__":
    main()