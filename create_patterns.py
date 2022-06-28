import numpy as np


def create_psp_template(width, height, frequency, number=4, horizontal=True):

    patterns = []
    phase_shifts = []

    phase_shift = 2 * np.pi / number
    
    length = width
    if not horizontal:
        length = height

    for i in range(number):
        phase_shifts.append(phase_shift * i)
        x = np.linspace(0, length, length)
        cos = np.zeros((1, length))
        cos[0, :] = np.cos(2 * np.pi * frequency * (x / length) - phase_shifts[i])
        pattern = np.zeros((height, width) if horizontal else (width, height))
        pattern[:] = cos
        patterns.append(pattern)

    return patterns, phase_shifts

def create_psp_templates(width, height, num_frequency_value, number=4, horizontal=True):
    patterns_with_diff_freq = []
    frequencies = []
    phase_shifts = []

    for i in range(num_frequency_value):
        frequencies.append(1 + 3 * i)
        template, phase_shifts = create_psp_template(width, height, frequencies[i])
        patterns_with_diff_freq.append(template)

    return patterns_with_diff_freq, phase_shifts, frequencies 


def linear_gradient(width, height):
    gradient = []
    for i in range(height):
        x = np.linspace(0, 255, width)
        gradient.append(x)
    return np.array(gradient, np.ubyte)