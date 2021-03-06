import errno
import os


def write_to_file_running_time(filename, content):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    fo = open(filename, "w")
    fo.write(content + "minutes")
    fo.close()


def write_to_file_running_steps(filename, content):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    fo = open(filename, "a")
    fo.write(content)
    fo.close()
