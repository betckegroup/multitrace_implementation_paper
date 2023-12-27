import numpy as np


def plane_wave(point, k_ext, polarization, direction):
    return polarization * np.exp(1j * k_ext * np.dot(point, direction))


def scaled_plane_wave(point, k_ext, polarization, direction):
    return plane_wave(point, k_ext, polarization, direction)


def tangential_trace(k_ext, polarization, direction):
    def tangential_trace(point, n, domain_index, result):
        result[:] = np.cross(
            scaled_plane_wave(point, k_ext, polarization, direction), n
        )

    return tangential_trace


def scaled_plane_wave_curl(point, k_ext, polarization, direction):
    return np.cross(direction, polarization) * np.exp(
        1j * k_ext * np.dot(point, direction)
    )


def neumann_trace(k_ext, polarization, direction):
    def neumann_trace(point, n, domain_index, result):
        result[:] = np.cross(
            scaled_plane_wave_curl(point, k_ext, polarization, direction), n
        )

    return neumann_trace
