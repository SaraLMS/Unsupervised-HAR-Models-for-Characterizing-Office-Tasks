import unittest
import os
import shutil
from synchronization.synchronization import _check_in_path


# Testing the private _check_in_path function directly due to its critical validation logic.
class TestCheckInPath(unittest.TestCase):
    def setUp(self):
        """Create a directory structure for testing."""
        self.test_dir = "test_data_check_in_path"
        os.makedirs(os.path.join(self.test_dir, "subfolder"), exist_ok=True)
        with open(os.path.join(self.test_dir, "subfolder", "test.txt"), 'w') as f:
            f.write("dummy content")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_valid_path(self):
        """Test a valid path containing subfolders with .txt files."""
        _check_in_path(self.test_dir)  # Should not raise an exception

    def test_invalid_path(self):
        """Test with a non-existent path."""
        with self.assertRaises(ValueError):
            _check_in_path("non_existent_path")

    def test_path_with_no_subfolders(self):
        """Test a path that has no subfolders."""
        shutil.rmtree(os.path.join(self.test_dir, "subfolder"))  # Remove subfolder
        with self.assertRaises(ValueError):
            _check_in_path(self.test_dir)

    def test_subfolder_with_no_txt_files(self):
        """Test a subfolder that contains no .txt files."""
        os.remove(os.path.join(self.test_dir, "subfolder", "test.txt"))  # Remove the .txt file
        with self.assertRaises(ValueError):
            _check_in_path(self.test_dir)


if __name__ == '__main__':
    unittest.main()
