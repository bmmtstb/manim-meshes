[![Python Test and Lint](https://github.com/bmmtstb/manim-meshes/actions/workflows/python_ci_test.yaml/badge.svg)](https://github.com/bmmtstb/manim-meshes/actions/workflows/python_ci_test.yaml)
# Manim for Meshes

> ⚠️ Work in progress
> 
> Most of the code will be rearranged or changed to use OpenGL, but OpenGL is not yet used throughout manim-ce. Stay tuned.

Manim-Trimeshes implements manim functionalities for different types of meshes using either basic node-face data structures or for importing the python [trimesh](https://pypi.org/project/trimesh/ "trimesh on pypi") library.

It is mainly developed as a Project for Interactive Graphics Systems Group (GRIS) at TU Darmstadt, but is publicly available for everyone interested in rendering and animating meshes.

## Installation

If published to pypi, can be installed using:

``pip install manim-meshes``

## Usage

``from manim_meshes import *``

[//]: #  (TODO create basic use-case with code)


## Example

[//]: # (TODO create working example + video)

In venv Run one of the minimal test examples: `manim tests/test_scene.py ConeScene`.
Multiple other examples can be found in the `tests/test_scene.py` file.


## Development
Set `./src/`-folder as project sources root and `./tests/`-folder as tests sources root if necessary.

Activate venv: `cd ./manim_meshes/`, then `poetry shell`

Install: `poetry install`

Update packages and .lock file: `poetry update`

If you implemented some features, update version using poetry: `poetry version prerelease|patch|minor|major`
See the Poetry [Documentation](https://python-poetry.org/docs/cli/#version).

Even though if CI works properly, Publish is automatically, it can be done manually with: `poetry publish --build`

[//]: # (TODO decide which git to use)
