"""
MIT License

Copyright (c) 2017 Cyrille Rossant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, tan, radians


# rendered image size
w = 1280
h = 720


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(ray_org, ray_dir, plane_point, plane_norm):
    # Return the distance from 'ray_org' to the intersection of the ray with the
    # plane, or +inf if there is no intersection.
    # 'ray_org' and 'plane_point' are 3D points, 'ray_dir' and 'plane_point' are normalized vectors.
    denominator = np.dot(ray_dir, plane_norm)
    if np.abs(denominator) < 1e-6:
        return np.inf
    d = np.dot(plane_point - ray_org, plane_norm) / denominator
    if d < 0:
        return np.inf
    return d


def intersect_sphere(ray_org, ray_dir, sphere_center, sphere_radius):
    # Return the distance from 'ray_org' to the intersection of the ray with the
    # sphere, or +inf if there is no intersection.
    # 'ray_org' and 'sphere_center' are 3D points, 'ray_dir' is a normalized vector, sphere_radius is a scalar.
    a = np.dot(ray_dir, ray_dir)
    center2org = ray_org - sphere_center
    b = 2 * np.dot(ray_dir, center2org)
    c = np.dot(center2org, center2org) - sphere_radius * sphere_radius
    disc = b * b - 4 * a * c
    if disc > 0:
        dist_sqrt = np.sqrt(disc)
        q = (-b - dist_sqrt) / 2.0 if b < 0 else (-b + dist_sqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect(ray_org, ray_dir, obj):
    if obj['type'] == 'plane':
        return intersect_plane(ray_org, ray_dir, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(ray_org, ray_dir, obj['position'], obj['radius'])


def get_normal(obj, obj_point):
    # Find normal.
    norm = np.zeros(3)
    if obj['type'] == 'sphere':
        norm = normalize(obj_point - obj['position'])
    elif obj['type'] == 'plane':
        norm = obj['normal']
    return norm


def get_color(obj, obj_point):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(obj_point)
    return color


def trace_ray(ray_org, ray_dir):
    # Find first point of intersection with the scene.
    t = np.inf
    obj_idx = 0
    for i, obj in enumerate(scene):
        t_obj = intersect(ray_org, ray_dir, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return

    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    obj_point = ray_org + ray_dir * t
    # Find properties of the object.
    normal = get_normal(obj, obj_point)
    color = get_color(obj, obj_point)
    to_light = normalize(L - obj_point)
    to_camera = normalize(camera_pos - obj_point)

    # Shadow: find if the point is shadowed or not.
    intersected_t = [intersect(obj_point + normal * .0001, to_light, obj_sh)
                     for k, obj_sh in enumerate(scene) if k != obj_idx]
    if intersected_t and min(intersected_t) < np.inf:
        return  # wicked not though ambient lighting bro

    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(normal, to_light), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(normal, normalize(to_light + to_camera)), 0) ** specular_k * color_light
    return obj, obj_point, normal, col_ray


def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color),
                reflection=0.5, refraction=0.5, ior=1.5)


def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda obj_point: (color_plane_0
                                         if (int(obj_point[0] * 2) % 2) == (int(obj_point[2] * 2) % 2)
                                         else color_plane_1),
                diffuse_c=0.75, specular_c=0.5, reflection=0.25, refraction=0, ior=1)


# List of objects.
color_plane_0 = np.ones(3)
color_plane_1 = np.zeros(3)
scene = [add_sphere([0.75, 0.1, 1.0], 0.6, [0.0, 0.0, 1.]),
         add_sphere([-0.75, 0.1, 2.25], 0.6, [0.5, 0.223, 0.5]),
         add_sphere([-2.75, 0.1, 3.5], 0.6, [1.0, 0.572, 0.184]),
         add_plane([0.0, -0.5, 0.0], [0.0, 1., 0.0])]

# Light position and color.
L = np.array([5.0, 5.0, -10.0])
color_light = np.ones(3)

# Default light and material parameters.
ambient = 0.05
diffuse_c = 1.0
specular_c = 1.0
specular_k = 50

depth_max = 2  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
background_col = (0, 0, 0)

# allocate image buffer
img = np.zeros((h, w, 3))

# define camera position and orientation
camera_pos = np.array([0.0, 0.35, -1.])  # Camera position
camera_look_at = np.array([0.0, 0.0, 0.0])  # Camera look at
camera_forward = normalize(camera_look_at - camera_pos)
camera_up = np.array([0.0, 1.0, 0.0])

# calculate camera right and exact camera up
camera_right = np.cross(camera_forward, camera_up)
camera_up = np.cross(camera_right, camera_forward)

# define projection plane
proj_dist = 1
fov = 90

aspect_ratio = float(h) / w
proj_w = 2 * tan(radians(fov / 2)) * proj_dist
proj_h = proj_w * aspect_ratio

proj_center = camera_pos + camera_forward * proj_dist
proj_lower_left = proj_center - camera_right * proj_w / 2 - camera_up * proj_h / 2
proj_upper_right = proj_center + camera_right * proj_w / 2 + camera_up * proj_h / 2


def fresnel(ray_dir, obj_norm, ior):
    cos_i = np.clip(np.dot(ray_dir, obj_norm), -1, 1)
    eta_i = 1
    eta_t = ior
    if cos_i > 0:
        eta_i, eta_t = eta_t, eta_i
    # Compute sin_е using Snell's law
    sin_t = eta_i / eta_t * sqrt(max(0., 1 - cos_i * cos_i))
    # Total internal reflection
    if sin_t >= 1:
        kr = 1
    else:
        cos_t = sqrt(max(0., 1 - sin_t * sin_t))
        cos_i = abs(cos_i)
        r_s = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t))
        r_p = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t))
        kr = (r_s * r_s + r_p * r_p) / 2
    return kr


def reflect(ray_dir, obj_norm):
    return ray_dir - 2 * np.dot(ray_dir, obj_norm) * obj_norm


def refract(ray_dir, obj_norm, ior):
    cos_i = np.clip(np.dot(ray_dir, obj_norm), -1, 1)
    eta_i = 1.
    eta_t = ior
    n = obj_norm

    if cos_i < 0:
        cos_i = -cos_i
    else:
        eta_i, eta_t = eta_t, eta_i
        n = -obj_norm

    eta = eta_i / eta_t
    k = 1 - eta * eta * (1 - cos_i * cos_i)
    if k < 0:
        return np.zeros((3,))
    return eta * ray_dir + (eta * cos_i - sqrt(k)) * n


def cast_ray(ray_org, ray_dir, depth=0):
    if depth > depth_max:
        return background_col

    traced = trace_ray(ray_org, ray_dir)
    if traced:
        obj, obj_point, obj_norm, col_ray = traced
        # compute fresnel
        kr = fresnel(ray_dir, obj_norm, obj.get("ior", 1))
        outside = np.dot(ray_dir, obj_norm) < 0
        bias = 0.001 * obj_norm
        # compute refraction if it is not a case of total internal reflection
        if kr < 1:
            refr_dir = refract(ray_dir, obj_norm, obj.get("ior", 1))
            refr_dir /= max(abs(refr_dir))
            refr_org = obj_point - bias if outside else obj_point + bias
            refr_col = cast_ray(refr_org, refr_dir, depth + 1)
        else:
            refr_col = np.zeros((3,))

        refl_dir = reflect(ray_dir, obj_norm)
        refl_dir /= max(abs(refl_dir))
        refl_org = obj_point + bias if outside else obj_point - bias
        refl_col = cast_ray(refl_org, refl_dir, depth + 1)

        # mix object color with reflected and refracted components
        hit_col = col_ray
        hit_col += np.multiply(obj.get("reflection", 1), refl_col)
        fresnel_col = np.multiply(kr, refl_col) + np.multiply(1 - kr, refr_col)
        hit_col += np.multiply(obj.get("refraction", 1), fresnel_col)
        return hit_col

    return background_col


# Loop through all pixels.
for i, x in enumerate(np.linspace(proj_lower_left[0], proj_upper_right[0], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")

    for j, y in enumerate(np.linspace(proj_lower_left[1], proj_upper_right[1], h)):
        col[:] = 0
        col_refl = np.zeros_like(col)
        col_refr = np.zeros_like(col)
        camera_look_at[:2] = (x, y)
        camera_forward = normalize(camera_look_at - camera_pos)

        reflection = 1.0
        refraction = 1.0

        # cast initial ray
        col += cast_ray(camera_pos, camera_forward)

        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('fig.png', img)
