# Manim for Meshes

> ⚠️ Work in progress
> 
> Most of the code will be rearranged or changed to use OpenGL, but OpenGL is not yet used throughout manim-ce. Stay tuned.

Manim-Trimeshes implements manim functionalities for different types of meshes using either basic node-edge data structures or the python [trimesh](https://pypi.org/project/trimesh/ "trimesh on pypi") library.

It is mainly developed as a Project for Interactive Graphics Systems Group (GRIS) at TU Darmstadt, but is publicly available for everyone interested in rendering and animating meshes.

## Installation

If published to pypi, can be installed using:

``pip install manim-meshes``

## Usage

``from manim_meshes import *``

[//]: #  (TODO create basic use-case with code)


## Example

[//]: # (TODO create working example + video)

In venv Run one of the minimal test examples: `manim tests/test_scene.py PyramidScene`.


## Development
Set `./src/`-folder as project sources root and `./tests/`-folder as tests sources root if necessary.

Activate venv: `cd ./manim_meshes/`, then `poetry shell`

Install: `poetry install`

Update packages and .lock file: `poetry update`

Publish: `poetry publish --build`

[//]: # (TODO decide which git to use)
