[![Python Test and Lint](https://github.com/bmmtstb/manim-meshes/actions/workflows/python_ci_test.yaml/badge.svg)](https://github.com/bmmtstb/manim-meshes/actions/workflows/python_ci_test.yaml)
# Manim for Meshes

> ⚠️ Work in progress
> 
> Most of the code will be rearranged or changed to use OpenGL, but OpenGL is not yet used throughout manim-ce. Stay tuned or feel free to assist.

Manim-Trimeshes implements manim functionalities for different types of meshes using either basic node-face data structures or by importing meshes from the python [trimesh](https://pypi.org/project/trimesh/ "trimesh on pypi") library.

It is mainly developed as a Project for Interactive Graphics Systems Group (GRIS) at TU Darmstadt, but is publicly available for everyone interested in rendering and animating meshes.

## Installation

Manim-meshes has been published to [pypi](https://pypi.org/project/manim-meshes/) and therefore can be easily installed using:

``pip install manim-meshes``

## Usage

Keep in mind this is a WIP...


``from manim_meshes import *``

The basic `ManimMesh` from `models` can currently only be used for smaller meshes (<1k Nodes), because it is dependent on the manim internal shaders which are not really implemented optimally. This type of mesh can be easily used for 2D and smaller 3D explanatory videos, not for high resolution rendering.
The more advanced `FastManimMesh` from `faster_models` uses a custom shader which needs to be inserted into the base manim implementation at this time! But therefore it can render enormous meshes.

[//]: #  (TODO create basic use-case with code)


## Example

[//]: # (TODO create working example + video)

In venv Run one of the minimal test examples: `manim tests/test_scene.py ConeScene`.
Multiple other examples can be found in the `tests/test_scene.py` file.


## Development
Set `./src/`-folder as project sources root and `./tests/`-folder as tests sources root if necessary.

Activate venv: `cd ./manim_meshes/`, then `poetry shell`

Install: `poetry install`
If you get errors, it is possible that you have to pip install `pycairo` and or `manimpango` manually, depending on your setup. Make sure to run `poetry install` until there are no more errors!

Update packages and your own .lock file: `poetry update`

If you implemented some features, update version using the matching poetry command: `poetry version prerelease|patch|minor|major`
See the Poetry [Documentation](https://python-poetry.org/docs/cli/#version).

Even though if the CI works properly, Publish on master branch is automatically, it can be done manually with: `poetry publish --build`
