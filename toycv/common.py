import functools
import hashlib
import os
import pickle
import re
import time
from collections import defaultdict
from os import path
from typing import Literal

import numpy
import torch


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


def equals_re(string_a, string_b, pattern_a='.*', pattern_b='.*'):
    """
    This function is used to compare two strings based on a regular expression pattern.
    The default pattern is '(.*)', which matches any part.

    :param string_a: The first string to compare.
    :param string_b: The second string to compare.
    :param pattern_a: The regular expression pattern to use for comparison. Default is '(.*)'.
    :param pattern_b: The regular expression pattern to use for comparison. Default is '(.*)'.
    :return: True if the strings match based on the pattern, False otherwise.
    """
    # print(string_a, string_b)
    # print(re.findall(pattern_a, str(string_a)), re.findall(pattern_b, str(string_b)))
    return re.findall(pattern_a, str(string_a)) == re.findall(pattern_b, str(string_b))


def join_re(list_a, list_b, key_a=0, key_b=0, how: Literal["inner", "left", "right", "outer"] = 'inner',
            pattern_a="(.*)", pattern_b="(.*)", keep_key=Literal["all", "a", "b"]):
    """
    This function is used to join two lists of tuples based on a common key.
    The function supports inner, left, right, and outer joins.
    The default join type is inner join.

    :param list_a: A list of tuples.
    :param list_b: Another list of tuples.
    :param key_a: The index of the key in list_a.
    :param key_b: The index of the key in list_b.
    :param how: The type of join to perform. Default is 'inner'.
    :param pattern_a: The regular expression pattern to use for comparison. Default is '(.*)'.
    :param pattern_b: The regular expression pattern to use for comparison. Default is '(.*)'.
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
                if equals_re(a[key_a], b[key_b], pattern_a, pattern_b):
                    if keep_key == "a":
                        result.append([*a, *b[:key_b], *b[key_b + 1:]])
                    elif keep_key == "b":
                        result.append([*a[:key_a], *a[key_a + 1:], *b])
                    else:
                        result.append([*a, *b])
    elif how == 'left':
        for a in list_a:
            for b in list_b:
                if equals_re(a[key_a], b[key_b], pattern_a, pattern_b):
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
                if equals_re(a[key_a], b[key_b], pattern_a, pattern_b):
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
                if equals_re(a[key_a], b[key_b], pattern_a, pattern_b):
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
                if equals_re(a[key_a], b[key_b], pattern_a, pattern_b):
                    break
            else:  # no break
                result.append([*(None,) * (len(list_a[0]) - 1), *b])

    return result


def group_re(file_paths, pattern="(.*)", listify=False):
    # 创建一个defaultdict来按照id进行分组
    grouped_files = defaultdict(list)

    # 按照id进行分组
    for file_path in file_paths:
        match = re.search(pattern, file_path)
        if match:
            id = match.group(1)
            grouped_files[id].append(file_path)

    if listify:
        grouped_files = list(grouped_files.items())
    else:
        grouped_files = dict(grouped_files)

    return grouped_files


def file_cache(cache_dir, hash=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hash of the function name and arguments to use as the cache filename
            if hash:
                cache_key = hashlib.md5(pickle.dumps((func.__name__, args, kwargs))).hexdigest()
                cache_path = path.join(cache_dir, f"{cache_key}.pkl")
            else:
                args_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', f"args={args}, kw={kwargs}")
                args_filename = re.sub(r'(^\.|(\s+\.|\.\s+)$)', '', args_filename)
                cache_path = path.join(cache_dir,
                                       f"{func.__name__}({args_filename}).pkl")

            # If cache file exists, read from it; otherwise, call the original function and save the result to the cache
            if path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
            else:
                result = func(*args, **kwargs)
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)

            return result

        return wrapper

    return decorator


# def extend_list_get_set(alist, get_method=lambda x: x, set_method=lambda x: x):
#     class ListWithMethod(list):
#         def __getitem__(self, item):
#             return get_method(alist[item])
#
#         def __setitem__(self, key, value):
#             alist[key] = set_method(value)
#
#     return ListWithMethod()


class extend_list_get_set(list):
    def __init__(self, alist, get_method=lambda x: x, set_method=lambda x: x):
        self.get_method = get_method
        self.set_method = set_method
        super().__init__(alist)

    def __getitem__(self, item):
        return self.get_method(super().__getitem__(item))

    def __setitem__(self, key, value):
        super().__setitem__(key, self.set_method(value))


def normalize(tensor: torch.Tensor):
    """
    Normalize tensor to [0, 1].
    :param tensor: torch.Tensor.
    :return: torch.Tensor.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    return (tensor - tensor_min) / (tensor_max - tensor_min)
