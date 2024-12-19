import unittest
import numpy as np
import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import apply_no_idle_constraint, apply_no_wait_constraint, apply_blocking_constraint, apply_johnson_rule, apply_cds_rule

class TestSchedulingAlgorithms(unittest.TestCase):
    def setUp(self):
        # Example processing times
        self.processing_times_2_machines = np.array([
            [2, 5, 9, 6, 8, 9, 3, 8, 5],  # Machine 1
            [3, 7, 4, 7, 6, 7, 6, 9, 3],  # Machine 2
        ])
        
        self.processing_times_3_machines = np.array([
            [4, 3, 5, 2, 7, 3, 6, 7, 5],  # Machine 1
            [8, 5, 2, 4, 3, 7, 6, 8, 9],  # Machine 2
            [3, 7, 4, 7, 5, 6, 6, 8, 3],  # Machine 3
        ])

    def test_johnson_rule(self):
        sequence = apply_johnson_rule(self.processing_times_2_machines)
        self.assertEqual(sequence, [1, 3, 7, 5, 9, 8, 4, 2, 6])  # Expected sequence

    def test_cds_rule(self):
        sequence = apply_cds_rule(self.processing_times_3_machines)
        self.assertIsInstance(sequence, list)
        self.assertEqual(len(sequence), 9)  # Should return a sequence of 9 jobs

    def test_no_idle_constraint(self):
        start_times, completion_times = apply_no_idle_constraint([0, 1], self.processing_times_2_machines)
        self.assertTrue(np.all(start_times >= 0))  # Start times should be non-negative

    def test_no_wait_constraint(self):
        start_times, completion_times = apply_no_wait_constraint([0, 1], self.processing_times_2_machines)
        self.assertTrue(np.all(start_times >= 0))  # Start times should be non-negative

    def test_blocking_constraint(self):
        start_times, completion_times = apply_blocking_constraint([0, 1], self.processing_times_2_machines)
        self.assertTrue(np.all(start_times >= 0))  # Start times should be non-negative

if __name__ == '__main__':
    unittest.main()
