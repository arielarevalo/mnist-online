from app.model import *


def test_validate_input_valid():
    # Test with valid input
    valid_input = [[[1 for _ in range(28)] for _ in range(28)] for _ in range(5)]  # 5 valid 28x28 matrices
    result = validate_input(valid_input)
    assert all(isinstance(m, np.ndarray) for m in result) and len(result) == 5, "Validation failed for valid input."


def test_validate_input_invalid_dimensions():
    # Test with invalid input (wrong dimensions)
    invalid_input_dimensions = [[[1 for _ in range(27)] for _ in range(28)] for _ in range(5)]  # Incorrect dimensions
    try:
        validate_input(invalid_input_dimensions)
        assert False, "Validation should fail for input with incorrect dimensions."
    except ValueError:
        pass  # Expected


def test_validate_input_invalid_values():
    # Test with invalid input (negative values)
    invalid_input_negative = \
        [[[1 if (i, j) != (0, 0) else -1 for i in range(28)] for j in range(28)] for _ in
         range(5)]  # Contains negative value
    try:
        validate_input(invalid_input_negative)
        assert False, "Validation should fail for input containing negative values."
    except ValueError:
        pass  # Expected
