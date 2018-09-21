(pyrieef) Python Riemannian Electrical Fields
=============

Pure python project to generate and learn collision free movement
in real environments using a Riemannian geometric approach.

Install

    pip install -r requirements.txt

Run all tests

    bash -c pytest tests

Result of

    python examples/plot_barrier.py

[![EF](https://s22.postimg.cc/bqln6ds2p/image.png)](https://postimg.cc/image/62fcfhnq5/)

For a technical presentation of the algorithm developed in this work
refer to the following publication

    @inproceedings{mainprice2016warping,
      title={Warping the workspace geometry with electric potentials for motion optimization of manipulation tasks},
      author={Mainprice, Jim and Ratliff, Nathan and Schaal, Stefan},
      booktitle={Intelligent Robots and Systems (IROS), IEEE/RSJ International Conference on},
      pages={3156--3163},
      year={2016}
    }
