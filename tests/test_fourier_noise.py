import unittest
import numpy as np
from hw2d.initializations.fourier_noise import get_fft_noise


class TestFFTNoise(unittest.TestCase):
    def test_fft_noise_output_shape(self):
        resolution = (256, 256)
        size = 1.0
        scale = 2.0
        noise = get_fft_noise(resolution, size, scale)

        # Test that output is numpy array
        self.assertIsInstance(noise, np.ndarray)

        # Test the output shape matches the given resolution
        self.assertEqual(noise.shape, resolution)

    def test_fft_noise_mean_std(self):
        resolution = (256, 256)
        size = 1.0
        scale = 2.0
        noise = get_fft_noise(resolution, size, scale)

        # Check if the noise mean is close to zero and standard deviation is close to 1
        self.assertTrue(np.isclose(np.mean(noise), 0.0, atol=1e-5))
        self.assertTrue(np.isclose(np.std(noise), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
