import unittest
import numpy as np
import pandas as pd
from src.domain.services.label_service import SimpleLabelService

class TestSimpleLabelService(unittest.TestCase):
    def setUp(self):
        self.class_names = ['apple', 'banana', 'cherry']
        self.service = SimpleLabelService(self.class_names)

    def test_initialization(self):
        self.assertEqual(self.service.classes, self.class_names)
        self.assertEqual(self.service._encoder['apple'], 0)
        self.assertEqual(self.service._decoder[1], 'banana')

    def test_transform_strings(self):
        labels = ['cherry', 'apple', 'banana']
        expected = np.array([2, 0, 1], dtype=np.int64)
        result = self.service.transform(labels)
        np.testing.assert_array_equal(result, expected)

    def test_transform_integers_passthrough(self):
        # Valid indices 0-2
        labels = [2, 0, 1]
        expected = np.array([2, 0, 1], dtype=np.int64)
        result = self.service.transform(labels)
        np.testing.assert_array_equal(result, expected)

    def test_transform_invalid_integers(self):
        # 9 is out of bounds for 3 classes, should default to 0
        labels = [2, 9, 1]
        expected = np.array([2, 0, 1], dtype=np.int64)
        result = self.service.transform(labels)
        np.testing.assert_array_equal(result, expected)

    def test_inverse_transform(self):
        indices = np.array([0, 2])
        expected = np.array(['apple', 'cherry'])
        result = self.service.inverse_transform(indices)
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        result = self.service.transform([])
        self.assertEqual(len(result), 0)
        
        result_inv = self.service.inverse_transform([])
        self.assertEqual(len(result_inv), 0)

    def test_single_input(self):
        self.assertEqual(self.service.transform('banana')[0], 1)
        self.assertEqual(self.service.inverse_transform(1)[0], 'banana')

if __name__ == '__main__':
    unittest.main()
