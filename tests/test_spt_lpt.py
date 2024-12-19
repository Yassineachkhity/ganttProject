import unittest
import numpy as np
import sys
import os
from flask import Flask

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

class TestSPTandLPT(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

        # Example processing times
        self.processing_times = [
            [4, 5, 6],
            [2, 3, 1],
            [5, 8, 7]
        ]

    def test_spt_rule(self):
        with self.client as c:
            response = c.post('/solve', json={
                "rule": "spt",
                "num_jobs": 3,
                "num_machines": 3,
                "processing_times": self.processing_times
            })
            data = response.get_json()
            self.assertIsNotNone(data)
            self.assertEqual(data['sequence'], [2, 1, 3])  # Expected sequence based on SPT

    def test_lpt_rule(self):
        with self.client as c:
            response = c.post('/solve', json={
                "rule": "lpt",
                "num_jobs": 3,
                "num_machines": 3,
                "processing_times": self.processing_times
            })
            data = response.get_json()
            self.assertIsNotNone(data)
            self.assertEqual(data['sequence'], [3, 1, 2])  # Expected sequence based on LPT

if __name__ == '__main__':
    unittest.main()
