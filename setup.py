#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 bhills <benjaminhhills@gmail.com>
# Distributed under terms of the GNU GPL3.0 license.

from setuptools import setup

version = '1.0.1'

packages = ['iceotherm',
            'iceotherm.lib']

requires = ['numpy',
            'matplotlib',
            'scipy']

setup(
    name='iceotherm',
    version=version,
    description='A Python package for glacier and ice-sheet thermal modeling',
    url='https://github.com/benhills/iceotherm',
    author='Benjamin Hills',
    author_email='bhills@uw.edu',
    license='GNU GPL-3.0',
    packages=packages,
    install_requires=requires
)
