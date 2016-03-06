import os

ROOT = os.path.dirname(__file__)
RESOURCES = os.path.join(ROOT, "resources")


def res(*paths):
    f = os.path.join(RESOURCES, *paths)
    if not os.path.exists(f):
        raise Exception("File path {abspath} does not exist.".format(abspath=os.path.abspath(f)))
    return f
