.. _getting_started:

Getting started
###############

.. _install_pyrieef_python:

Installing from PyPI
=============================

Pyrieef Python packages are distributed via
`PyPI <https://pypi.org/project/pyrieef/>`_.

Supported Python versions:

* 3.6
* 3.7
* 3.8

Supported operating systems:

* Ubuntu 18.04+
* macOS 10.14+

If you have other Python versions (e.g. Python 2) or operating systems, please
refer to :ref:`compilation` and compile Open3D from source.

Pip (PyPI)
----------

.. code-block:: bash

    pip install pyrieef

.. note::
    In general, we recommend using a
    `virtual environment <https://docs.python-guide.org/dev/virtualenvs/>`_ for
    containerization. Otherwise, depending on the configurations, ``pip3`` may
    be needed for Python 3, or the ``--user`` option may need to be used to
    avoid permission issues. For example:

    .. code-block:: bash

        pip3 install pyrieef
        # or
        pip install --user pyrieef
        # or
        python3 -m pip install --user pyrieef

Try it
------

Now, try importing Open3D.

.. code-block:: bash

    python -c "import pyrieef"

If this works, congratulations, now pyrieef has been successfully installed!
