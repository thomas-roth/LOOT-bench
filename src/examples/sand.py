import genesis as gs
import numpy as np
import torch

gs.init(
    # logging_level='warning',
    # performance_mode=True, faster simulation but slower compiliation -> only activate for policy training
    # backend=gs.cpu, #gs.gpu # does select atomatically
)


def init_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -1.0, 0.0),
            upper_bound=(0.5, 1.0, 1),
            #particle_size=8e-3,
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    # with relative path also search genesis/assets
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
    scene.add_entity(gs.morphs.Box(
        size = (0.08, 0.08, 0.08),
        pos = (0.65, 0.0, 0.13),
    ))

    # elastic object
    scene.add_entity(
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Box(
            pos=(0.0, -0.5, 0.25),
            size=(0.2, 0.2, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='visual',
        ),
    )

    # sand object
    scene.add_entity(
        material=gs.materials.MPM.Sand(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
        ),
    )

    # plastic object
    scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.5, 0.35),
            radius=0.1,
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.3, 1.0),
        ),
    )

    scene.add_entity(
        gs.morphs.MJCF(file="xml/cable.xml"),
    )

    cam = scene.add_camera(res=(1280, 720), pos=(2, 0, 2), lookat=(0, 0, 0.5))

    scene.build()
    return scene, robot, cam


def sample_random_action(robot):
    lows = []
    highs = []
    for joint in robot.joints:
        if joint.n_dofs > 0:
            lower, upper = joint.dofs_limit[0]

            if isinstance(lower, torch.Tensor):
                lower = lower.detach().cpu().numpy()
            if isinstance(upper, torch.Tensor):
                upper = upper.detach().cpu().numpy()
            lower = np.atleast_1d(lower)
            upper = np.atleast_1d(upper)

            lows.append(lower)
            highs.append(upper)
    lows = np.concatenate(lows)
    highs = np.concatenate(highs)

    action = np.random.uniform(lows, highs, size=robot.n_dofs)
    return action


if __name__ == "__main__":
    scene, robot, cam = init_scene()

    cam.start_recording()

    for _ in range(1000):
        action = sample_random_action(robot)

        robot.control_dofs_position(action)

        scene.step()
        cam.render()

    cam.stop_recording(save_to_filename="video.mp4", fps=60)
