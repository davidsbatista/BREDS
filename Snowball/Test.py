#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

from Pattern import Pattern
from Tuple import Tuple


def main():

    t = Tuple('E1', 'E2', 'teste', None, None, None, None)
    t1 = Tuple('E1', 'E4', 'teste1', None, None, None, None)
    t2 = Tuple('E2', 'E5', 'teste2', None, None, None, None)
    t3 = Tuple('E3', 'E6', 'teste3', None, None, None, None)

    l = [t1, t2, t3]

    if t in l:
        print "True"
    else:
        print "False"

    p1 = Pattern(t)
    p2 = Pattern(t)

    l = [p1, p2]

    p3 = Pattern(t)

    if p3 not in l:
        print "not there"
        l.append(p3)


if __name__ == "__main__":
    main()
