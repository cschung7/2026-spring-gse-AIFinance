from termcolor import colored
import numpy as np
import pandas as pd


TITLE = "PYTHON / NUMPY / PANDAS BASICS"


def print_section(title: str) -> None:
    line = "-" * 60
    print(colored(f"\n{line}\n{title}\n{line}", "cyan"))


def python_basics() -> None:
    print_section("1. PYTHON BASICS")

    # variables and types
    x_int = 10
    x_float = 3.14
    x_str = "hello"
    x_list = [1, 2, 3]
    x_dict = {"name": "Alice", "age": 30}

    print(colored("Variables and types:", "green"))
    print(f"x_int={x_int} type={type(x_int)}")
    print(f"x_float={x_float} type={type(x_float)}")
    print(f"x_str={x_str} type={type(x_str)}")
    print(f"x_list={x_list} type={type(x_list)}")
    print(f"x_dict={x_dict} type={type(x_dict)}")

    # control flow
    print(colored("\nIf / for / while:", "green"))
    if x_int > 5:
        print("x_int is greater than 5")

    for n in x_list:
        print(f"for-loop value: {n}")

    counter = 0
    while counter < 3:
        print(f"while-loop counter: {counter}")
        counter += 1

    # functions
    print(colored("\nFunctions:", "green"))

    def add(a, b):
        return a + b

    print(f"add(2, 3) = {add(2, 3)}")


def numpy_basics() -> None:
    print_section("2. NUMPY BASICS")

    arr = np.array([1, 2, 3, 4, 5])
    print(colored("1D array:", "green"))
    print("arr =", arr)
    print("arr dtype:", arr.dtype)
    print("arr shape:", arr.shape)
    print("arr mean:", arr.mean())
    print("arr sum:", arr.sum())

    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print(colored("\n2D array (matrix):", "green"))
    print(matrix)
    print("shape:", matrix.shape)
    print("row 0:", matrix[0])
    print("element (1, 2):", matrix[1, 2])

    print(colored("\nVectorized operations:", "green"))
    print("arr + 10 ->", arr + 10)
    print("arr * 2 ->", arr * 2)
    print("arr ** 2 ->", arr ** 2)


def pandas_basics() -> None:
    print_section("3. PANDAS BASICS")

    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 40],
        "score": [88.5, 92.0, 79.5, 95.0],
    }
    df = pd.DataFrame(data)

    print(colored("DataFrame:", "green"))
    print(df)

    print(colored("\nHead / describe:", "green"))
    print(df.head())
    print(df.describe())

    print(colored("\nSelect column:", "green"))
    print("ages:\n", df["age"])
    print("mean age:", df["age"].mean())

    print(colored("\nFiltering:", "green"))
    print("age > 30:\n", df[df["age"] > 30])

    print(colored("\nNew column:", "green"))
    df["age_group"] = np.where(df["age"] >= 30, "30+", "<30")
    print(df)


def main() -> None:
    print(colored(TITLE, "yellow", attrs=["bold"]))
    python_basics()
    numpy_basics()
    pandas_basics()


if __name__ == "__main__":
    main()

