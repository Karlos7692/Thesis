import numpy as np

def insert2d(insert_array, array2d, i):
    return np.insert(array2d, i, insert_array, axis=1)


def insert3d(insert_array, array3d, i):
    return np.insert(array3d, i, insert_array, axis=2)

def multiple_insert3d(insert_array, array3d, indexes):
    for i in indexes:
        array3d = insert3d(insert_array, array3d, i)
    return array3d

