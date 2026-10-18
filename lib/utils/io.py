"""Point-cloud, mesh and image visualization writers."""
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from torch import Tensor


_SMPL_COLORS_RGB = (
    (230, 30, 30), (30, 200, 30), (30, 80, 230), (220, 190, 20),
    (190, 40, 210), (30, 180, 220), (230, 60, 120), (160, 210, 80),
)


def write_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype(np.uint8)
    if len(points) != len(colors):
        raise ValueError(f"Point/color count mismatch: {len(points)} vs {len(colors)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            handle.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces + 1:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")


def write_combined_obj(path: Path, vertices: Sequence[np.ndarray], faces: np.ndarray) -> None:
    """Write per-person RGB materials shared with the reprojection palette."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    material_path = path.with_suffix(".mtl")
    with material_path.open("w", encoding="ascii", newline="\n") as material:
        for person_index in range(len(vertices)):
            rgb = np.asarray(_SMPL_COLORS_RGB[person_index % len(_SMPL_COLORS_RGB)]) / 255.0
            material.write(f"newmtl person_{person_index:02d}\n")
            material.write(f"Kd {rgb[0]:.8f} {rgb[1]:.8f} {rgb[2]:.8f}\n")
            material.write("Ka 0 0 0\nKs 0 0 0\nd 1\nillum 1\n\n")
    offset = 0
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"mtllib {material_path.name}\n")
        for person_index, person_vertices in enumerate(vertices):
            handle.write(f"o person_{person_index:02d}\n")
            handle.write(f"usemtl person_{person_index:02d}\n")
            person_vertices = np.asarray(person_vertices, dtype=np.float32).reshape(-1, 3)
            for vertex in person_vertices:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in faces + 1 + offset:
                handle.write(f"f {face[0]} {face[1]} {face[2]}\n")
            offset += len(person_vertices)


def save_parameters(
    path: Path, prediction: Dict[str, Tensor], labels: List[int]
) -> None:
    values = dict(
        instance_labels=np.asarray(labels, dtype=np.int64),
        pose_axis_angle=prediction["pose_axis_angle"][:, 0].float().cpu().numpy(),
        rotation_matrices=prediction["rotation_matrices"][:, 0].float().cpu().numpy(),
        betas=prediction["betas"][:, 0].float().cpu().numpy(),
        translation=prediction["translation"][:, 0].float().cpu().numpy(),
        scale=prediction["scale"][:, 0].float().cpu().numpy(),
        person_center=prediction["person_center"][:, 0].float().cpu().numpy(),
        observed_point_scale=prediction["person_scale"][:, 0].float().cpu().numpy(),
        core_point_scale=prediction["core_scale"][:, 0].float().cpu().numpy(),
        core_fraction=prediction["core_fraction"][:, 0].float().cpu().numpy(),
        expanded_count=prediction["expanded_count"][:, 0].float().cpu().numpy(),
        actual_voxel_size=prediction["actual_voxel_size"][:, 0].float().cpu().numpy(),
        joints_local=prediction["joints_local"][:, 0].float().cpu().numpy(),
        joints_world=prediction["joints_world"][:, 0].float().cpu().numpy(),
        point_count=prediction["point_count"][:, 0].cpu().numpy(),
        view_entropy=prediction["mean_view_entropy"][:, 0].float().cpu().numpy(),
    )
    for key in (
        "romp_init_rotation_matrices", "delta_rotation_matrices",
        "translation_residual",
    ):
        if key in prediction:
            values[key] = prediction[key][:, 0].float().cpu().numpy()
    np.savez_compressed(path, **values)


def _clip_near_plane(triangle: np.ndarray, near: float) -> List[np.ndarray]:
    """Clip one camera-space triangle at z=near; return a triangle fan."""
    polygon = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
        start_inside, end_inside = start[2] >= near, end[2] >= near
        if start_inside:
            polygon.append(start)
        if start_inside != end_inside:
            weight = (near - start[2]) / (end[2] - start[2])
            polygon.append(start + weight * (end - start))
    return [np.asarray([polygon[0], polygon[i], polygon[i + 1]])
            for i in range(1, len(polygon) - 1)]


def _rasterize_triangle(
    uv: np.ndarray, depth: np.ndarray, color: Sequence[int],
    pixels: np.ndarray, z_buffer: np.ndarray,
) -> None:
    """Fill pixel centers using perspective-correct camera depth, in-place."""
    height, width = z_buffer.shape
    x_min, y_min = np.maximum(np.floor(uv.min(axis=0)), 0).astype(np.int64)
    x_max, y_max = np.minimum(np.ceil(uv.max(axis=0)), [width - 1, height - 1]).astype(
        np.int64
    )
    if x_min > x_max or y_min > y_max:
        return
    x0, y0 = uv[0]
    x1, y1 = uv[1]
    x2, y2 = uv[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-12:
        return
    yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1] + 0.5
    w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
    w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
    inverse_depth = w0 / depth[0] + w1 / depth[1] + w2 / depth[2]
    triangle_depth = np.full(w0.shape, np.inf)
    np.divide(1.0, inverse_depth, out=triangle_depth, where=inside & (inverse_depth > 0))
    current_depth = z_buffer[y_min:y_max + 1, x_min:x_max + 1]
    visible = inside & (triangle_depth < current_depth)
    current_depth[visible] = triangle_depth[visible]
    pixels[y_min:y_max + 1, x_min:x_max + 1][visible] = color


def _render_smpl_meshes(
    vertices: np.ndarray, faces: np.ndarray, c2w: np.ndarray,
    intrinsic_pixel: np.ndarray, image_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """World-space [P,N,3] meshes -> uint8 RGB and [H,W] coverage mask.

    Camera convention: +z forward, +x right, +y down. All people share a
    z-buffer; triangle winding is unrestricted. No scene-point occlusion.
    """
    height, width = image_shape
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf)
    world_to_camera = np.linalg.inv(c2w)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for person_index, person_vertices in enumerate(vertices):
        camera = person_vertices @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        color = _SMPL_COLORS_RGB[person_index % len(_SMPL_COLORS_RGB)]
        for face in faces:
            triangle = camera[face]
            if not np.isfinite(triangle).all():
                continue
            triangles = [triangle] if (triangle[:, 2] >= 1e-6).all() else (
                _clip_near_plane(triangle, 1e-6)
            )
            for clipped in triangles:
                projected = clipped @ intrinsic_pixel.T
                uv = projected[:, :2] / projected[:, 2:3]
                _rasterize_triangle(uv, clipped[:, 2], color, pixels, z_buffer)
    return pixels, np.isfinite(z_buffer)


def save_smpl_reprojections(
    output_dir: Path, images: Tensor, paths: Sequence[Path],
    prediction: Dict[str, Tensor], c2w: np.ndarray,
    intrinsic_pixel: np.ndarray, faces: np.ndarray, overlay_alpha: float,
) -> List[str]:
    """Save one mesh-on-RGB overlay per view using the OBJ material colors."""
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1")
    reprojection_dir = output_dir / "smpl_reprojections"
    reprojection_dir.mkdir(parents=True, exist_ok=True)
    vertices = prediction["vertices_world"][:, 0].detach().float().cpu().numpy()
    saved: List[str] = []
    for view in range(images.shape[1]):
        rgb = (
            images[0, view].detach().float().clamp(0, 1)
            .permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().astype(np.uint8)
        mesh_rgb, mask = _render_smpl_meshes(
            vertices, faces, c2w[view], intrinsic_pixel[view], rgb.shape[:2]
        )
        overlay = rgb.copy()
        overlay[mask] = np.rint(
            (1.0 - overlay_alpha) * rgb[mask] + overlay_alpha * mesh_rgb[mask]
        ).astype(np.uint8)
        stem = paths[view].stem.replace(" ", "_")
        overlay_path = reprojection_dir / f"view_{view:02d}_{stem}_overlay.png"
        if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to save {overlay_path}")
        saved.append(str(overlay_path))
    return saved
