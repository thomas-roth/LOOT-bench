import genesis as gs
import genesis.vis.keybindings as kb

import time
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.vis.viewer_plugins.plugins.mouse_interaction import MouseInteractionPlugin


gs.init()

def init_scene():
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True
    )

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'), )
    cube = scene.add_entity(gs.morphs.Box(
        size=(0.08, 0.08, 0.08),
        pos=(0.65, 0.0, 0.13),
    ))

    scene.viewer.add_plugin(gs.vis.viewer_plugins.MouseInteractionPlugin())
    plugin = ImGuiOverlayPlugin()
    scene.viewer.add_plugin(plugin)

    scene.build()

    return scene, plugin



if __name__ == "__main__":
    scene, plugin = init_scene()
    plugin.paused = True

    frame = 0
    while scene.viewer.is_alive():
        if plugin.rebuild_requested:
            scene.destroy()
            scene, plugin = init_scene()
        if plugin.should_step():
            scene.step()
        frame += 1
        time.sleep(0.01)

    is_running = True
