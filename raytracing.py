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
from math import sqrt

from numba.roc.hsadrv.drvapi import hsa_amd_image_descriptor_t

w = 1280
h = 720


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])


def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return

    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(L - M)
    toO = normalize(O - M)

    # Shadow: find if the point is shadowed or not.
    l = [intersect(M + N * .0001, toL, obj_sh) 
            for k, obj_sh in enumerate(scene) if k != obj_idx]
    if l and min(l) < np.inf:
        return  # wicked not though ambient lighting bro

    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, M, N, col_ray


def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
                radius=np.array(radius), color=np.array(color),
                reflection=.5, refraction=0.5, ior=1.5)


def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
                normal=np.array(normal),
                color=lambda M: (color_plane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2)
                                 else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=.25, refraction=0, ior=1)


# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
# scene = [add_sphere([0.0, .1, 0.30], .6, [0., 0., 1.]),
#          add_sphere([-0.7, .1, 1.2], .6, [.5, .223, .5]),
#          # add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
#          add_plane([0., -.5, 0.], [0., 1., 0.]),
#         ]
scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
         add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
         add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
    ]

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 2  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera pointing to.
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)
background_col = (0, 0, 0)


def fresnel(I, N, ior):
    cos_i = np.clip(np.dot(I, N), -1, 1)
    eta_i = 1
    eta_t = ior
    if cos_i > 0:
        eta_i, eta_t = eta_t, eta_i
    # Compute sini using Snell's law
    sin_t = eta_i / eta_t * sqrt(max(0., 1 - cos_i * cos_i))
    # Total internal reflection
    if sin_t >= 1:
        kr = 1
    else:
        cost = sqrt(max(0., 1 - sin_t * sin_t))
        cos_i = abs(cos_i)
        Rs = ((eta_t * cos_i) - (eta_i * cost)) / ((eta_t * cos_i) + (eta_i * cost))
        Rp = ((eta_i * cos_i) - (eta_t * cost)) / ((eta_i * cos_i) + (eta_t * cost))
        kr = (Rs * Rs + Rp * Rp) / 2
    return kr


def reflect(I, N):
    return I - 2 * np.dot(I, N) * N


def refract(I, N, ior):
    cos_i = np.clip(np.dot(I, N), -1, 1)
    eta_i = 1.
    eta_t = ior
    n = N

    if cos_i < 0:
        cos_i = -cos_i
    else:
        eta_i, eta_t = eta_t, eta_i
        n = -N

    eta = eta_i / eta_t
    k = 1 - eta * eta * (1 - cos_i * cos_i)
    if k < 0:
        return np.zeros((3,))
    return eta * I + (eta * cos_i - sqrt(k)) * n


def cast_ray(orig, dir, depth=0):
    if depth > depth_max:
        return background_col

    traced = trace_ray(orig, dir)
    if traced:
        obj, M, N, col_ray = traced
        # compute fresnel
        kr = fresnel(dir, N, 1.5)
        outside = np.dot(dir, N) < 0
        bias = 0.001 * N
        # compute refraction if it is not a case of total internal reflection
        if kr < 1:
            refr_dir = refract(dir, N, obj.get("ior", 1))
            refr_dir /= max(abs(refr_dir))
            refr_org = M - bias if outside else M + bias
            refr_col = cast_ray(refr_org, refr_dir, depth + 1)
        else:
            refr_col = np.zeros((3,))

        refl_dir = reflect(dir, N)
        refl_dir /= max(abs(refl_dir))
        refl_org = M + bias if outside else M - bias
        refl_col = cast_ray(refl_org, refl_dir, depth + 1)

        # mix object color with reflected and refracted components
        hit_col = col_ray
        hit_col += np.multiply(obj.get("reflection", 1), refl_col)
        fresnel_col = np.multiply(kr, refl_col) + np.multiply(1 - kr, refr_col)
        hit_col += np.multiply(obj.get("refraction", 1), fresnel_col)
        return hit_col

    return background_col


# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        col_refl = np.zeros_like(col)
        col_refr = np.zeros_like(col)
        Q[:2] = (x, y)
        D = normalize(Q - O)
        rayO, rayD = O, D

        reflection = 1.0
        refraction = 1.0

        # cast initial ray
        col += cast_ray(rayO, rayD)

        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('fig.png', img)
