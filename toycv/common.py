import re
import time

import numpy


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


def equals_re(string_a, string_b, pattern='.*'):
    """
    This function is used to compare two strings based on a regular expression pattern.
    The default pattern is '(.*)', which matches any part.

    :param string_a: The first string to compare.
    :param string_b: The second string to compare.
    :param pattern: The regular expression pattern to use for comparison. Default is '(.*)'.
    :return: True if the strings match based on the pattern, False otherwise.
    """
    # print(string_a, string_b)
    # print(re.findall(pattern, str(string_a)), re.findall(pattern, str(string_b)))
    return re.findall(pattern, str(string_a)) == re.findall(pattern, str(string_b))


def join_re(list_a, list_b, key_a=0, key_b=0, how='inner', pattern="(.*)", keep_key="all"):
    """
    This function is used to join two lists of tuples based on a common key.
    The function supports inner, left, right, and outer joins.
    The default join type is inner join.

    :param list_a: A list of tuples.
    :param list_b: Another list of tuples.
    :param key_a: The index of the key in list_a.
    :param key_b: The index of the key in list_b.
    :param how: The type of join to perform. Default is 'inner'.
    :param pattern: The regular expression pattern to use for comparison. Default is '(.*)'.
    :param keep_key: The key to keep when joining. Default is 'a'.
    :return: A list of tuples that are the result of the join operation.
    """
    result = []
    if not (isinstance(list_a[0], tuple) or isinstance(list_a[0], list) or isinstance(list_a[0], numpy.ndarray)):
        list_a = [(a,) for a in list_a]

    # list_a = tuple(list_a)

    if not (isinstance(list_b[0], tuple) or isinstance(list_b[0], list) or isinstance(list_b[0], numpy.ndarray)):
        list_b = [(b,) for b in list_b]

    # list_b = tuple(list_b)

    # print(list_a)
    # print(list_b)

    if how == 'inner':
        for a in list_a:
            for b in list_b:
                if equals_re(a[key_a], b[key_b], pattern):
                    if keep_key == "a":
                        result.append([*a, *b[:key_b], *b[key_b + 1:]])
                    elif keep_key == "b":
                        result.append([*a[:key_a], *a[key_a + 1:], *b])
                    else:
                        result.append([*a, *b])
    elif how == 'left':
        for a in list_a:
            for b in list_b:
                if equals_re(a[key_a], b[key_b], pattern):
                    if keep_key == "a":
                        result.append([*a, *b[:key_b], *b[key_b + 1:]])
                    elif keep_key == "b":
                        result.append([*a[:key_a], *a[key_a + 1:], *b])
                    else:
                        result.append([*a, *b])
                    break
            else:  # no break
                result.append([*a, *(None,) * (len(list_b[0]) - 1)])
    elif how == 'right':
        for b in list_b:
            for a in list_a:
                if equals_re(a[key_a], b[key_b], pattern):
                    if keep_key == "a":
                        result.append([*a, *b[:key_b], *b[key_b + 1:]])
                    elif keep_key == "b":
                        result.append([*a[:key_a], *a[key_a + 1:], *b])
                    else:
                        result.append([*a, *b])
                    break
            else:  # no break
                result.append([*(None,) * len(list_a[0]), *b])

    elif how == 'outer':
        for a in list_a:
            for b in list_b:
                if equals_re(a[key_a], b[key_b], pattern):
                    if keep_key == "a":
                        result.append([*a, *b[:key_b], *b[key_b + 1:]])
                    elif keep_key == "b":
                        result.append([*a[:key_a], *a[key_a + 1:], *b])
                    else:
                        result.append([*a, *b])
                    break
            else:  # no break
                result.append([*a, *(None,) * (len(list_b[0]) - 1)])

        for b in list_b:
            for a in list_a:
                if equals_re(a[key_a], b[key_b], pattern):
                    break
            else:  # no break
                result.append([*(None,) * (len(list_a[0]) - 1), *b])

    return result
