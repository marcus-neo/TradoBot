#!/bin/sh

pylint src2/ && flake8 src2 && pydocstyle --add-select=D203,D212,D205,D200 --add-ignore=D211 --match='(?!__init__).*\.py' src2/