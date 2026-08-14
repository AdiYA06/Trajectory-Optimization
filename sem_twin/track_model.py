import numpy as np

def latlong_to_xy(lat, lon, lat0, lon0):
    R = 6371000  # Earth radius, metres
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y

def _straight_segment(length_m: float, ds: float, v_limit: float = 20.0):
    """Build a straight segment of the track, sampled at ds metres."""
    n = int(length_m / ds)
    s = np.arange(n) * ds
    gradient = np.zeros(n)
    curvature = np.zeros(n)
    v_limit = np.full(n, v_limit)
    return s, gradient, curvature, v_limit

def _hill_segment(length_m: float, peak_grad: float, 
                  ds: float, v_limit: float = 20.0, ramp_len: float = 10.0):
    n = int(length_m / ds)
    s = np.arange(n) * ds
    gradient = np.full(n, peak_grad)

    ramp_n = int(ramp_len / ds)
    ramp_in = peak_grad * (1 - np.cos(np.linspace(0, np.pi/2, ramp_n))) / 2
    gradient[:ramp_n] = ramp_in
    gradient[-ramp_n:] = ramp_in[::-1]

    curvature = np.zeros(n)
    v_limit = np.full(n, v_limit)
    return s, gradient, curvature, v_limit

def _corner_segment(radius_m: float, arc_angle: float, ds: float, v_limit_corner: float = 10.0):
    arc_length = radius_m * arc_angle
    n = int(arc_length / ds)
    s = np.arange(n) * ds
    gradient = np.zeros(n)
    curvature = np.full(n, 1.0 / radius_m)
    v_limit = np.full(n, v_limit_corner)
    return s, gradient, curvature, v_limit

class Track:
    def __init__(self, s: np.ndarray, gradient: np.ndarray,
                 curvature: np.ndarray, v_limit: np.ndarray):
        self.s = s                  # distance along track, metres, increasing
        self.gradient = gradient    # road angle theta(s), radians (+ = uphill)
        self.curvature = curvature  # 1/radius(s), 1/m (0 = straight)
        self.v_limit = v_limit      # externally-imposed speed limit at s, m/s
                                    # (placeholder constant for now; later
                                    # replaced by the bicycle-model limit)

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    def gradient_at(self, s_check: float) -> float:
        """Interpolate gradient at arbitrary distance s."""
        grad_s = np.interp(s_check, self.s, self.gradient)
        return grad_s
        # — decide on boundary behaviour (clip vs extrapolate vs raise).

    def curvature_at(self, s_check: float) -> float:
        """TODO: same pattern as gradient_at."""
        curv_s = np.interp(s_check, self.s, self.curvature)
        return curv_s

    def v_limit_at(self, s_check: float) -> float:
        """TODO: same pattern as gradient_at."""
        v_lim_s = np.interp(s_check, self.s, self.v_limit)
        return v_lim_s

    @classmethod
    def make_synthetic_track(cls, ds=1.0) -> "Track":
        """Build the starter track: straight -> uphill -> downhill -> corners
        -> start/finish straight."""
        segments = [
        _straight_segment(200, ds),
        _hill_segment(150, peak_grad=0.02, ds=ds),
        _hill_segment(150, peak_grad=-0.02, ds=ds),
        _corner_segment(radius_m=15, arc_angle=np.pi / 2, ds=ds),
        _corner_segment(radius_m=15, arc_angle=np.pi / 2, ds=ds),
        _straight_segment(100, ds),
        ]

        s_parts, grad_parts, curv_parts, vlim_parts = [], [], [], []
        running_s = 0.0

        for s_local, gradient, curvature, v_limit in segments:
            s_parts.append(s_local + running_s)
            grad_parts.append(gradient)
            curv_parts.append(curvature)
            vlim_parts.append(v_limit)
            running_s += s_local[-1] + ds   # move the running total past this segment

        s = np.concatenate(s_parts)
        gradient = np.concatenate(grad_parts)
        curvature = np.concatenate(curv_parts)
        v_limit = np.concatenate(vlim_parts)

        return cls(s=s, gradient=gradient, curvature=curvature, v_limit=v_limit)

    @classmethod
    def from_gpx(cls, filepath: str) -> "Track":
        """Load a real track from a GPX file (use `gpxpy`)."""
        import gpxpy
        with open(filepath, "r") as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        points = [(p.latitude, p.longitude, p.elevation)
                   for track in gpx.tracks for seg in track.segments for p in seg.points]
        lat0, long0 = points[0][0], points[0][1]
        xy = np.array([latlong_to_xy(lat, long, lat0, long0) for lat, long, _ in points])
        elevations = np.array([elev for _, _, elev in points])

        # min_segment_m = 0.01   # anything closer than 1cm apart is GPS noise, not real motion
        # keep_mask = np.concatenate([[True], segment_lengths > min_segment_m])

        # xy = xy[keep_mask]
        # elevations = elevations[keep_mask]

        dx, dy = np.diff(xy[:, 0]), np.diff(xy[:, 1])
        segment_lengths = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(segment_lengths)))

        d_elev = np.diff(elevations)
        gradient = np.concatenate(([0], np.arctan2(d_elev, segment_lengths)))

        headings = np.arctan2(dy, dx)
        d_heading = np.diff(headings)
        d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        curvature = np.concatenate(([[0.0], [0.0], d_heading / segment_lengths[1:]]))

        v_limit = np.full(len(s), 15.0) # placeholder limit

        ds = 1.0
        s_uniform = np.arange(0.0, s[-1], ds)
        gradient_uniform = np.interp(s_uniform, s, gradient)
        curvature_uniform = np.interp(s_uniform, s, curvature)
        v_limit_uniform = np.full(len(s_uniform), 15.0)

        return cls(s=s_uniform, gradient=gradient_uniform, curvature=curvature_uniform, v_limit=v_limit_uniform)