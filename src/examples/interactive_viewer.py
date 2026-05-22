import importlib.util
import time
from pathlib import Path

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.vis.viewer_plugins.plugins.mouse_interaction import MouseInteractionPlugin

gs.init()


class InteractiveViewer:
    def __init__(self, scene_file: str, sleep_s=0.01):
        self.scene_file = scene_file
        self.sleep_s = sleep_s
        self.scene_builder = None
        self.scene = None
        self.plugin = None

    def _load_scene_builder(self):
        module_path = Path(__file__).with_name(self.scene_file)
        spec = importlib.util.spec_from_file_location(
            "interactive_viewer_scene", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load scene module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def build_scene(self):
        self.scene_builder = self._load_scene_builder()
        self.scene = self.scene_builder.add_entities()

        # add interactive plugins before building the scene
        self.scene.viewer.add_plugin(MouseInteractionPlugin(spring_const=100.0))
        self.plugin = ImGuiOverlayPlugin()
        self.plugin.paused = True
        self.scene.viewer.add_plugin(self.plugin)

        self.scene.build()

    def rebuild_scene(self):
        if self.scene is not None:
            self.scene.destroy()
        self.build_scene()

    def step_once(self):
        if self.scene is not None:
            self.scene.step()

    def run(self):
        if self.scene is None:
            self.build_scene()

        while self.scene.viewer.is_alive():
            if getattr(self.plugin, "rebuild_requested", False):
                self.rebuild_scene()
            if callable(getattr(self.plugin, "should_step", None)):
                if self.plugin.should_step():
                    self.scene.step()
            else:
                self.scene.step()
            time.sleep(self.sleep_s)
