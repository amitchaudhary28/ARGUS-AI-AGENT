
def calculate_total(price: float, quantity: int) -> float:
    """Calculate the total price."""
    return price * quantity


def get_discount(total: float, discount_rate: float) -> float:
    """Calculate the discounted amount."""
    if not 0 <= discount_rate <= 1:
        raise ValueError("Discount rate must be between 0 and 1.")

    return total * (1 - discount_rate)


def main() -> None:
    total = calculate_total(100.0, 3)
    discount = get_discount(total, 0.1)
    print(f"Final amount: {total - discount:.2f}")


if __name__ == "__main__":
    main()
