# env_balls_pymunk.py
import math
import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

import pymunk


class PymunkBallWorldSimulator:
    """
    2D ball world using pymunk (Chipmunk2D) to mimic the paper's matter-js setup:

    - World size: 800 x 600
    - Ball radius: 60 px
    - Max initial speed: 60 px per step
    - Elastic ball-ball and ball-wall collisions
    - Optional variable mass: m in {1, 5, 25}

    Internal state is in pixel units; output is normalized:
      x_norm = x / width
      y_norm = y / width   (paper divides both x,y by width)
      vx_norm = vx / v_max
      vy_norm = vy / v_max
      mass is kept unnormalized.
    """

    def __init__(
        self,
        width: float = 800.0,
        height: float = 600.0,
        ball_radius: float = 60.0,
        v_max: float = 60.0,
        dt: float = 0.1,
        num_balls: int = 4,
        variable_mass: bool = False,
        mass_values=(1.0, 5.0, 25.0),
        seed: Optional[int] = None,
    ):
        self.width = float(width)
        self.height = float(height)
        self.ball_radius = float(ball_radius)
        self.v_max = float(v_max)
        self.dt = float(dt)
        self.num_balls = num_balls
        self.variable_mass = variable_mass
        self.mass_values = tuple(float(m) for m in mass_values)

        if seed is not None:
            random.seed(seed)

        self._build_static_world()

    # --------- build walls / static geometry --------- #

    def _build_static_world(self):
        """
        Create a pymunk Space with 4 walls forming a box.
        """
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        static_body = self.space.static_body
        w, h = self.width, self.height

        # Four box walls as segments
        walls = [
            pymunk.Segment(static_body, (0.0, 0.0), (w, 0.0), 0.0),        # bottom
            pymunk.Segment(static_body, (0.0, h), (w, h), 0.0),            # top
            pymunk.Segment(static_body, (0.0, 0.0), (0.0, h), 0.0),        # left
            pymunk.Segment(static_body, (w, 0.0), (w, h), 0.0),            # right
        ]
        for wall in walls:
            wall.elasticity = 1.0
            wall.friction = 0.0
        self.space.add(*walls)

    # --------- sampling helpers --------- #

    def _sample_non_overlapping_positions(self):
        positions = []
        r = self.ball_radius
        for _ in range(self.num_balls):
            for _ in range(10000):
                x = random.uniform(r, self.width - r)
                y = random.uniform(r, self.height - r)
                ok = True
                for (px, py) in positions:
                    if math.hypot(x - px, y - py) < 2.0 * r:
                        ok = False
                        break
                if ok:
                    positions.append((x, y))
                    break
            else:
                raise RuntimeError("Could not place all balls without overlap")
        return positions

    def _reset_balls(self):
        """
        Remove existing dynamic bodies and create new ones with random
        positions, velocities, and masses.
        """
        # Remove previous balls if they exist (keep static walls)
        for s in list(self.space.shapes):
            if isinstance(s, pymunk.Circle) and s.body is not self.space.static_body:
                self.space.remove(s, s.body)

        positions = self._sample_non_overlapping_positions()
        self.bodies = []

        for (x, y) in positions:
            mass = random.choice(self.mass_values) if self.variable_mass else 1.0
            radius = self.ball_radius
            moment = pymunk.moment_for_circle(mass, 0.0, radius)

            body = pymunk.Body(mass, moment)
            body.position = (x, y)

            # Sample random initial velocity, |v| <= v_max
            angle = random.uniform(0.0, 2.0 * math.pi)
            speed = random.uniform(0.0, self.v_max)
            body.velocity = (speed * math.cos(angle), speed * math.sin(angle))

            shape = pymunk.Circle(body, radius)
            shape.elasticity = 1.0
            shape.friction = 0.0

            self.space.add(body, shape)
            self.bodies.append((body, mass))

    # --------- main simulation --------- #

    def simulate(self, steps: int = 60) -> torch.Tensor:
        """
        Roll out a single trajectory using pymunk.

        Returns
        -------
        states : torch.Tensor of shape [T, N, 5]
            Each state = (x_norm, y_norm, vx_norm, vy_norm, mass).
        """
        self._reset_balls()
        T = steps
        N = self.num_balls

        states = torch.zeros(T, N, 5, dtype=torch.float32)
        w = self.width
        vmax = self.v_max

        for t in range(T):
            # record normalized state
            for i, (body, mass) in enumerate(self.bodies):
                x, y = body.position
                vx, vy = body.velocity
                x_norm = x / w
                y_norm = y / w  # NOTE: both divided by width, as in paper
                vx_norm = vx / vmax
                vy_norm = vy / vmax
                states[t, i] = torch.tensor(
                    [x_norm, y_norm, vx_norm, vy_norm, mass],
                    dtype=torch.float32,
                )

            if t == T - 1:
                break

            # advance physics
            self.space.step(self.dt)

        return states


class WindowDataset(Dataset):
    """
    Simple dataset wrapper over [num_samples, 3, N, D] tensor of windows.
    """

    def __init__(self, windows: torch.Tensor):
        self.windows = windows

    def __len__(self) -> int:
        return self.windows.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]
