import importlib.util
import time
from pathlib import Path

import genesis as gs


gs.init()

FILE_NAME = "interactive_viewer_scene.py"


def load_scene_builder(file_name:str):
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(file_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scene module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    scene_builder = load_scene_builder(FILE_NAME)
    scene, plugin = scene_builder.init_scene()

    while scene.viewer.is_alive():
        if plugin.rebuild_requested:
            scene.destroy()
            scene_builder = load_scene_builder(FILE_NAME)
            scene, plugin = scene_builder.init_scene()
        if plugin.should_step():
            scene.step()
        time.sleep(0.01)
