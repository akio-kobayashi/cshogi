# cshogi: A Fast Shogi Library for Python

cshogi is a fast Python shogi library that provides board management, legal move generation, move verification, USI protocol, and support for machine learning formats.

## Installation

You can install cshogi from PyPI if a pre-compiled wheel is available for your platform:

```bash
pip install cshogi
```

If you want to use the web interface, you can install the extra dependencies:
```bash
pip install cshogi[web]
```

## Building from Source (Linux/Ubuntu)

If a pre-compiled wheel is not available for your system, or if you want to build from the latest source, you can build it manually. This project uses `setuptools` and `Cython` to compile the C++ extensions.

**1. Install Prerequisites**

You will need a C++ compiler, Python development headers, and pip.

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

**2. Clone the Repository**

```bash
git clone https://github.com/TadaoYamaoka/cshogi.git
cd cshogi
```

**3. Install Build Dependencies**

Install the necessary Python packages to build the extension.

```bash
pip install cython numpy
```

**4. Build and Install cshogi**

Now, you can compile and install the library using pip.

```bash
pip install .
```
This command will use `setup.py` to build the Cython extensions and install the package into your Python environment.

## Features

*   Support for Python 3.6+ and Cython 0.29+.
*   Integration with IPython/Jupyter Notebook for board rendering.
*   Push/pop moves.
*   Text-based board representation.
*   Check, game over, and draw detection (including repetition and Nyūgyoku).
*   Handles moves in USI or CSA format.
*   Ability to read and write compressed position formats used by Apery and YaneuraOu.
*   Control USI-compliant engines.

## Acknowledgements

cshogi uses source code from `Apery` for many of its high-performance features.

## License

cshogi is licensed under the GPLv3. See the `LICENSE` file for details.
