#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

from Pattern import Pattern
from Tuple import Tuple


def main():

    t = Tuple('E1', 'E2', 'teste', None, None, None, None)

    p1 = Pattern(t)
    p2 = Pattern(t)

    l = [p1, p2]

    p3 = Pattern(t)

    if p3 not in l:
        l.append(p3)

    print l

if __name__ == "__main__":
    main()
