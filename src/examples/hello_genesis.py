import genesis as gs
import torch
import numpy as np

gs.init()


def init_scene():
    scene = gs.Scene(show_viewer=True)

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'), )
    cube = scene.add_entity(gs.morphs.Box(
        size=(0.08, 0.08, 0.08),
        pos=(0.65, 0.0, 0.13),
    ))

    cam = scene.add_camera(res=(1280, 720), pos=(2, 0, 2), lookat=(0, 0, 0))
    scene.build()
    return scene, robot, cam


def sample_random_action(robot):
    lows = []
    highs = []
    for joint in robot.joints:
        if joint.n_dofs > 0:
            lower, upper = joint.dofs_limit[0]

            # normalize to 1D to handle 1-DOF & multi-DOF joints
            if not isinstance(lower, torch.Tensor):
                lower = torch.tensor([lower], device=gs.device)
                upper = torch.tensor([upper], device=gs.device)
            elif lower.dim() == 0:
                lower = lower.unsqueeze(0)
                upper = upper.unsqueeze(0)

            lows.append(lower)
            highs.append(upper)
    lows = torch.cat(lows).cpu().numpy()
    highs = torch.cat(highs).cpu().numpy()

    action = np.random.uniform(lows, highs, size=robot.n_dofs)
    return action


if __name__ == "__main__":
    scene, robot, cam = init_scene()

    cam.start_recording()

    for i in range(100):
        action = sample_random_action(robot)
        robot.control_dofs_position(action)

        scene.step()
        cam.render()

    cam.stop_recording()
