import math
import unittest
from bark_ml.library_wrappers.lib_tf2rl.trajectories_compare import *

class TrajectoriesCompareTest(unittest.TestCase):

    def test_key_not_in_trajectory(self):
        """Test: Expection risen when a key is not in the trajectory.
        """
        first = {
            'obs': [[0] * 16] * 10,
            'act': [[2] * 2] * 10,
            }
        second = {
            'obs': [[1] * 16] * 10,
            'next_obs': [[2] * 16] * 10,
            }
        with self.assertRaises(ValueError):
            compare_trajectories(first, second)

    def test_compare_trajectories_same_dimensions(self):
        """Calculates the norm between two trajectories of the same length.
        """
        first = {
            'obs': [[1] * 16] * 10,
            'next_obs': [[2] * 16] * 10,
            'act': [[4] * 2] * 10,
            }
        second = {
            'obs': [[0] * 16] * 10,
            'next_obs': [[1] * 16] * 10,
            'act': [[2] * 2] * 10,
            }
        distances = compare_trajectories(first, second)
        self.assertAlmostEqual(distances['obs'], 4.0)
        self.assertAlmostEqual(distances['next_obs'], 4.0)
        self.assertAlmostEqual(distances['act'], math.sqrt(8))

    def test_compare_trajectories_different_dimensions(self):
        """Calculates the norm between two trajectories of different lengths.
        """
        first = {
            'obs': [[1] * 16] * 10,
            'next_obs': [[2] * 16] * 10,
            'act': [[4] * 2] * 10,
            }
        second = {
            'obs': [[0] * 16] * 8,
            'next_obs': [[1] * 16] * 8,
            'act': [[2] * 2] * 8,
            }
        distances = [compare_trajectories(first, second), compare_trajectories(second, first)]
        self.assertAlmostEqual(distances[0], distances[1])
        self.assertAlmostEqual(distances[0]['obs'], 8.0)
        self.assertAlmostEqual(distances[0]['next_obs'], 12.0)
        self.assertAlmostEqual(distances[0]['act'], math.sqrt(32) + math.sqrt(8))

    def test_compare_same_trajectories(self):
        """Calculates the norm between two equal trajectories.
        """
        first = {
            'obs': [[1] * 16] * 10,
            'next_obs': [[2] * 16] * 10,
            'act': [[4] * 2] * 10,
            }
        second = {
            'obs': [[1] * 16] * 10,
            'next_obs': [[2] * 16] * 10,
            'act': [[4] * 2] * 10,
            }
        distances = compare_trajectories(first, second)
        self.assertAlmostEqual(distances['obs'], 0.0)
        self.assertAlmostEqual(distances['next_obs'], 0.0)
        self.assertAlmostEqual(distances['act'], 0)

    def test_compare_same_trajectories(self):
        """Calculates the norm between two equal trajectories except one is longer.
        """
        first = {
            'obs': [[1] * 16] * 10,
            'next_obs': [[2] * 16] * 10,
            'act': [[4] * 2] * 10,
            }
        second = {
            'obs': [[1] * 16] * 8,
            'next_obs': [[2] * 16] * 8,
            'act': [[4] * 2] * 8,
            }
        distances = compare_trajectories(first, second)
        self.assertAlmostEqual(distances['obs'], 4.0)
        self.assertAlmostEqual(distances['next_obs'], 8.0)
        self.assertAlmostEqual(distances['act'], math.sqrt(32))

        

if __name__ == '__main__':
    unittest.main()