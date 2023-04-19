def blocks(files, size=65536):
    """
    Read the file block-wise and then count the '\n' characters in each block.
    """
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer
