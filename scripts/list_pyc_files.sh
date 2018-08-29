#!/bin/bash
find .. -name '*.pyc' -or -name '*.so' > pyc_files.txt
cat pyc_files.txt
