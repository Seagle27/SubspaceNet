""" This script is a help source for bgu to translate text files to arrays and for doing the
other direction too.
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def save_arrays_to_text(file_path, *arrays):
    """
    save multiple numpy arrays to a text file as floats
    :param file_path: the path to the output text file
    :param arrays: any number of numpy arrays to be saved to the file.
    """
    with open(file_path, 'w') as f:
        for arr in arrays:
            if np.any(np.iscomplex(arr)):
                real_part = arr.real
                imag_part = arr.imag
                combined = np.concatenate((real_part, imag_part), axis=0)
                np.savetxt(f, combined, fmt='%.9f')
            else:
                np.savetxt(f, arr, fmt='%.8f')
            f.write("\n")


def load_arrays_from_txt(file_path: str):
    """
    load multiple numpy arrays from a text file.
    Assumes arrays are separated by newlines nad are not complex.
    :param file_path: path to the text file to read
    :return: a tuple of numpy arrays loaded from the file.
    """
    arrays = []
    with open(file_path, 'r') as f:
        array_lines = f.read().strip().split('\n\n')

        for array_str in array_lines:
            array = np.genfromtxt(array_str.splitlines(), delimiter=' ', dtype=float)
            arrays.append(array)
    return tuple(arrays)


def load_complex_arrays_from_text(file_path, N=7):
    arrays = []
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n\n')

        for block in lines:
            data = np.genfromtxt(block.splitlines(), delimiter=' ')
            if data.shape[0] == 2 * N:
                real_part = data[:N, :]
                imag_part = data[N:2*N, :]
                complex_array = real_part +1j * imag_part
                arrays.append(complex_array)
            else:
                arrays.append(data)
    return tuple(arrays) if len(arrays) > 1 else arrays[0]


def test_arrays(file_path, *arrays):
    """
    tests that the number of arrays in the file matches the number of input arrays.
    also checks that the arrays in the file are equal
    :param file_path: the path to the text file
    :param arrays: the input arrays to be compared with the file's arrays
    """
    loaded_arrays = load_arrays_from_txt(file_path)
    # test if the number of arrays matches:
    assert len(loaded_arrays) == len(arrays),\
        f"Test failed: expected {len(arrays)}, but found {len(loaded_arrays)} arrays."
    # test if the arrays are equal:
    for i, (arr1, arr2) in enumerate(zip(loaded_arrays, arrays)):
        if arr1.shape != arr2.shape:
            arr1 = arr1.reshape(arr2.shape)
        assert np.allclose(arr1, arr2, 1e-8, 1e-8),\
            f"test failed: arrays at index {i} do not match"
    print('all test passed!')


def get_interp_data(az_array_to_interp, file_path_to_base_data):
    """
    :param az_array_to_interp: array of directions(azimuth).
     within the range of -100, 100, not necessarily integers.
    :param file_path_to_base_data: path to txt file that holds the initial basic data
    :return: array of shape: input_length, 2, 4 that represents the azimuth, abs, phase
    """
    all_arrays = load_arrays_from_txt(file_path_to_base_data)
    azimuth_base_array = all_arrays[0]
    amps_array = all_arrays[2]
    phase_array = all_arrays[1]
    assert np.all(amps_array.shape == phase_array.shape)
    # creating the interpolation:
    array_to_return = np.zeros(shape=(az_array_to_interp.shape[0], 2, amps_array.shape[1]))
    interp_object_phase =\
        interpolate.interp1d(azimuth_base_array, phase_array, axis=0,
                             bounds_error=False, fill_value=(phase_array[0, :], phase_array[-1, :]))
    interp_object_amps = (
        interpolate.interp1d(azimuth_base_array, amps_array, axis=0,
                             bounds_error=False, fill_value=(amps_array[0, :], amps_array[-1, :])))
    phase_for_dataset = interp_object_phase(az_array_to_interp)
    amps_for_dataset = interp_object_amps(az_array_to_interp)
    array_to_return[:, 0, :] = amps_for_dataset
    array_to_return[:, 1, :] = phase_for_dataset
    return array_to_return


def test_interpolation(size, path_to_txt, seed=9091):
    """tests random interpolation points.
    basic array is the initial x axes that was used for the interpolation.
    must be monotonically increasing"""
    np.random.seed(seed)
    x_to_test = np.random.uniform(-100, 99, size)
    x_to_test = np.round(x_to_test, 3)
    y_to_test = get_interp_data(x_to_test, path_to_txt)

    all_arrays = load_complex_arrays_from_text(path_to_txt)
    basic_azimuth = all_arrays[0]
    basic_amp = all_arrays[2]
    basic_phase = all_arrays[1]
    for i, x in enumerate(x_to_test):
        right_index = np.argmax((basic_azimuth - x) > 0)  # neighbor_index
        left_index = right_index - 1
        x0 = basic_azimuth[left_index]
        x1 = basic_azimuth[right_index]
        amp_y0 = basic_amp[left_index, :]
        amp_y1 = basic_amp[right_index, :]
        phase_y0 = basic_phase[left_index, :]
        phase_y1 = basic_phase[right_index, :]
        true_value_amp = amp_y0 + ((x - x0) * (amp_y1 - amp_y0) / (x1 - x0))
        true_value_phase = phase_y0 + ((x - x0) * (phase_y1 - phase_y0) / (x1 - x0))
        assert np.allclose(true_value_amp, y_to_test[i, 0, :], atol=1e-9, rtol=0),\
            f'test interpolation on mid point failed: test {i} out of {size}'
        assert np.allclose(true_value_phase, y_to_test[i, 1, :], atol=1e-9, rtol=0),\
            f'test interpolation on mid point failed: test {i} out of {size}'
    print(f'test interpolation on mid point passed all tests')


if __name__ == '__main__':
    a = load_arrays_from_txt('arrays.txt')
    angles = a[1] % 360
    angles = np.where(angles > 180, angles - 360, angles)
    plt.figure()
    plt.plot(angles[:,0], label="Antenna 1")
    plt.plot(angles[:,1] , label="Antenna 2")
    plt.plot(angles[:,2] , label="Antenna 3")
    plt.plot(angles[:,3] , label="Antenna 4")
    plt.legend()
    plt.show()

