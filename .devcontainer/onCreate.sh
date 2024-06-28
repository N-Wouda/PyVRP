#!/bin/bash

set -e

git submodule update --init

poetry install --with examples,docs,dev && rm -rf build/
poetry run pre-commit install
