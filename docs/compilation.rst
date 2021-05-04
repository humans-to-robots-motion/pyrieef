.. _compilation:

Build from source
=====================

.. _compiler_version:

System requirements
-------------------

* Ubuntu 18.04+: pip
* macOS 10.14+: pip
* Windows 10 (64-bit): pip

Cloning pyrieef
--------------

.. code-block:: bash

    git clone https://github.com/humans-to-robots-motion/pyrieef.git

.. _compilation_unix:

Ubuntu/macOS
------------

.. _compilation_unix_dependencies:

1. Setup Python environments
````````````````````````````

Activate the python ``virtualenv`` or Conda ``virtualenv```. Check
``which python`` to ensure that it shows the desired Python executable.
Alternatively, set the CMake flag ``-DPYTHON_EXECUTABLE=/path/to/python``
to specify the python executable.

If Python binding is not needed, you can turn it off by ``-DBUILD_PYTHON_MODULE=OFF``.

.. _compilation_unix_config:

2. Install dependencies
```````````````````````
.. code-block:: bash

    pip install -r requirements.txt


.. _compilation_unix_build:

3. Build
````````

TODO

.. _compilation_unix_install:

5. Install
``````````

TODO

Finally, verify the python installation with:

.. code-block:: bash

    python -c "import pyrieef"


Unit test
---------

To run Python unit tests:

.. code-block:: bash

    # Activate virtualenv first
    pip install pytest
    pytest tests