def format_number(num):
    """
    Convert a large number into a shorter format,
    making it easier to read. Formats into thousands (k),
    millions (M), billions (B), etc.

    Args:
    num (int): The number to format.

    Returns:
    str: The formatted number.
    """
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num // 1000}k"
    elif num < 1000000000:
        return f"{num // 1000000}M"
    elif num < 1000000000000:
        return f"{num // 1000000000}B"
    else:
        return f"{num // 1000000000000}T"
