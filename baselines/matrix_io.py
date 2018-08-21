import numpy as np
import struct
import sys
from datetime import datetime 


def ReadMatrixFromFile(file_name):
    with open(file_name, mode='rb') as f:
        content = f.read()
    row, col = struct.unpack('=ii', content[:8])
    data = struct.unpack('=' + 'f' * ((len(content) - 8) // 4), content[8:])
    m = np.array(data)
    m = np.reshape(m, (row, col), 'F')

    return m.astype(np.float32)

def WriteMatrixToFile(file_name, matrix):
    # First, get the row and col of the array.
    shape = matrix.shape
    if len(shape) == 1:
      row, col = 1, shape[0]
    else:
      row, col= matrix.shape
    row_bytes = row.to_bytes(4, sys.byteorder, signed=True)
    col_bytes = col.to_bytes(4, sys.byteorder, signed=True)
    # Second, get the raw data.
    data_bytes = matrix.astype(np.float32, 'F').tobytes('F')
    # Finally, write the binary file.
    f = open(file_name, 'w+b')
    content = b"".join([row_bytes, col_bytes, data_bytes])
    f.write(content)
    f.close()
