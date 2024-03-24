import time


def flatten(o):
    """
    This function is used to flatten a nested iterable (like a list or tuple) into a single, one-dimensional iterable.
    It uses Python's generator syntax and can handle any depth of nesting, including cases where the iterable contains strings.

    :param o: An iterable that can be nested to any depth. For example, [1, [2, [3, 4], 5]].
    :return: A generator that yields each item in the original iterable, in the order they appear, but flattened to a single level.
    """

    for item in o:
        if isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def get_current_timestamp():
    timestamp = time.time()
    return timestamp
