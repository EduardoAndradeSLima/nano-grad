import unittest
from nanograd.engine import Value


class TestEngine(unittest.TestCase):
    def test_value_creation(self):
        a = Value(4.0, label="a")
        self.assertEqual(a.data, 4.0)
        self.assertEqual(a.grad, 0.0)
        self.assertEqual(a.label, "a")
        self.assertEqual(repr(a), "Value(data=4.0, grad=0.0)")


if __name__ == "__main__":
    unittest.main()
