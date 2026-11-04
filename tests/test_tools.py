from tools import add, divide, multiply


def test_add() -> None:
    assert add.invoke({"a": 3, "b": 4}) == 7


def test_multiply() -> None:
    assert multiply.invoke({"a": 3, "b": 4}) == 12


def test_divide() -> None:
    assert divide.invoke({"a": 8, "b": 2}) == 4


def test_divide_by_zero() -> None:
    try:
        divide.invoke({"a": 8, "b": 0})
    except ValueError as error:
        assert str(error) == "Cannot divide by zero."
    else:
        raise AssertionError("divide should reject a zero denominator")
