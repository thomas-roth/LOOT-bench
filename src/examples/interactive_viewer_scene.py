import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin


def init_scene():
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(
        gs.morphs.Box(
            size=(0.08, 0.08, 0.08),
            pos=(0.65, 0.0, 0.13),
        )
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.5, 0.5, 0.5),
        )
    )


    #scene.viewer.add_plugin(gs.vis.viewer_plugins.MouseInteractionPlugin())
    plugin = ImGuiOverlayPlugin()
    plugin.paused = True
    scene.viewer.add_plugin(plugin)

    scene.build()

    return scene, plugin

