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

While executing a commandline manim script, make sure to set the `--renderer=opengl` flag, the Cairo renderer will mostly not work.

The basic `ManimMesh` and `Manim2DMesh` from `manim_models/basic_mesh` can currently only be used for smaller meshes (<1k Nodes), because it is dependent on the manim internal shaders which are not really implemented optimally. This type of mesh can be easily used for 2D and smaller 3D explanatory videos, not for high resolution rendering.
The more advanced `FastManimMesh` from `opengl_mesh` uses a custom shader which needs to be inserted into the base manim implementation at this time! But therefore it can render enormous meshes fast.
The `TriangleManim2DMesh` from `triangle_mesh` implements further functions that are only reasonable for triangle meshes. (e.g. Delaunay)

All these Mesh-Renders are based on the `Mesh`-Class, in `data_models`, which should implement a multitude of basic Mesh-functions. If you have the feeling something is missing, feel free to add it.

[//]: #  (TODO create basic use-case with code)


## Example

[//]: # (TODO create working example + video)

With active poetry venv Run one of the minimal test examples: `manim tests/test_scene.py ConeScene`.
Multiple other examples can be found in the `tests/test_scene.py` file.


## Development
In PyCharm set `./src/`-folder as project sources root and `./tests/`-folder as tests sources root if necessary.

Activate the poetry venv: `cd ./manim_meshes/`, then `poetry shell`

Install: `poetry install`
If you get errors, it is possible that you have to pip install `pycairo` and or `manimpango` manually (globally?), depending on your setup. Make sure to run `poetry install` until there are no more errors!

Update packages and your own .lock file: `poetry update`

If you implemented some features, update version using the matching poetry command: `poetry version prerelease|patch|minor|major`
See the Poetry [Documentation](https://python-poetry.org/docs/cli/#version).

Even though if the CI works properly, Publishing to pypi on master branch is automatically, it can be done manually with: `poetry publish --build`

### Debugging
Like with basic manim, create an executable Python file with something around:

```python
from tests.test_scene import SnapToGridScene
if __name__ == "__main__":
    scene = SnapToGridScene()
    scene.render()
```

Then debug the file and place breakpoints as expected. May not work with the "renderer=opengl" flag that is necessary for some scripts.