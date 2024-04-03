import unittest

from synchronization.synchronization import _check_supported_sensors, _check_acc_sensor_selected


class TestCheckSupportedSensors(unittest.TestCase):
    def test_valid_sensors(self):
        """Test with a valid selection of sensors."""
        valid_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc', 'wearheartrate']}
        try:
            _check_supported_sensors(valid_sensors)  # Should not raise an exception
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_invalid_device(self):
        """Test with an invalid device in the selection."""
        invalid_sensors = {'fridge': ['acc']}  # Assuming 'fridge' is not a supported device
        with self.assertRaises(ValueError):
            _check_supported_sensors(invalid_sensors)

    def test_invalid_sensor(self):
        """Test with an invalid sensor for a supported device."""
        invalid_sensors = {'phone': ['acc', 'x-ray']}  # Assuming 'x-ray' is not a supported sensor
        with self.assertRaises(ValueError):
            _check_supported_sensors(invalid_sensors)

    def test_missing_acc_sensor(self):
        """Test that an error is raised if 'acc' sensor is not selected for each device."""
        sensors_missing_acc = {'phone': ['gyr'], 'watch': ['gyr', 'wearheartrate']}
        with self.assertRaises(ValueError):
            _check_acc_sensor_selected(sensors_missing_acc)

    def test_acc_sensor_present_for_all_devices(self):
        """Test that no error is raised if 'acc' sensor is selected for all devices."""
        sensors_with_acc = {'phone': ['acc', 'gyr'], 'watch': ['acc', 'wearheartrate']}
        try:
            _check_acc_sensor_selected(sensors_with_acc)  # Should not raise an exception
        except ValueError as e:
            self.fail(f"Unexpected ValueError raised when 'acc' sensor was selected for all devices: {e}")


if __name__ == '__main__':
    unittest.main()
