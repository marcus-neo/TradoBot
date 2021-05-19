#!/bin/sh

pylint src/ && flake8 src/ && pydocstyle --add-selct=D203,D212,D205,D200 --add-ignore=D211 --match='(?!__init__).*\.py' src/