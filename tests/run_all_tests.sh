#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR
clear
bash -c "python -m pytest --disable-pytest-warnings"
# bash -c "pytest"
