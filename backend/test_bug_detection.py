def divide(a, b):
    if b == 0:
        return 0
    return a / b


def calculate():
    result = divide(10, 0)
    return result


if __name__ == "__main__":
    print(calculate())