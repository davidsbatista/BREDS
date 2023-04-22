from typing import TextIO, Generator


def blocks(files: TextIO, size: int = 65536) -> Generator[str, None, None]:
    """
    Read the file block-wise and then count the '\n' characters in each block.
    """
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer
