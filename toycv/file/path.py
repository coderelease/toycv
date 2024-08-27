import glob
import os
import queue
import re


def ensure_dirs(file_path: str, file_has_suffix: str = True) -> str:
    if file_has_suffix and "." in os.path.basename(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path

    directory = os.path.abspath(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return file_path


def home_path(*sub_dirs):
    return ensure_dirs(os.path.join(os.path.expanduser('~'), *sub_dirs))


def dir_file_ext(file_path):
    return os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0], os.path.splitext(file_path)[1]


def glob_re(root_dir, path_re="**"):
    """
    Match file paths based on regular expressions.

    :param root_dir: The root directory to start the search.
    :param path_re: The regular expression pattern to match the file paths. Default is "**" which matches all files.
    :return: A set of file paths that match the regular expression.
    """
    path_re_parts = path_re.split(os.path.sep)

    dir_queue = queue.SimpleQueue()
    dir_queue.put([root_dir, 0])
    result_set = set()

    while not dir_queue.empty():
        current_dir, parts_index = dir_queue.get()
        pending_paths = []

        if path_re_parts[parts_index] == "**":
            pending_paths = glob.glob(pathname=os.path.join(current_dir, "**"), recursive=True)
        else:
            pattern = re.compile(path_re_parts[parts_index])
            for listed_path in os.listdir(current_dir):
                if pattern.fullmatch(listed_path):
                    pending_paths.append(os.path.join(current_dir, listed_path))

        if parts_index == len(path_re_parts) - 1:
            result_set.update(pending_paths)
        else:
            for path in filter(os.path.isdir, pending_paths):
                dir_queue.put([path, parts_index + 1])

    # pprint(result_set)
    return result_set
