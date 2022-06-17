# Manim for Meshes

!! Work in progress !!

Manim-Trimeshes implements manim functionalities for triangular meshes.

It was developed as a Project for Fraunhofer-GRIS at TU-Darmstadt, but is publicly available for everyone.

## Installation

If published to pypi, can be installed using:

``pip install manim-trimeshes``

## Usage

``from manim_trimeshes import *``

[//]: #  (TODO create basic use-case with code)


## Example

[//]: # (TODO create working example + video)

In venv Run one of the minimal test examples: `manim tests/test_scene.py PyramidScene`


## Development
Set `./src/`-folder as project sources root and `./tests/`-folder as tests sources root if necessary.

Activate venv: `cd ./manim_trimeshes/`, then `poetry shell`

Install: `poetry install`

Update packages and .lock file: `poetry update`

Publish: `poetry publish --build`

[//]: # (TODO decide which git to use)
