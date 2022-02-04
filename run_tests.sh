#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run python tests
read -p "Execute python tests? press [Enter] or [n] : " RUN_PYTHON_TEST
if [[ -z $RUN_PYTHON_TEST ]]; then
    clear
    cd ${SCRIPT_DIR}/tests
    bash -c "python -m pytest --disable-pytest-warnings"
fi
echo "Done tests."
