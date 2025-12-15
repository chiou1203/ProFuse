import torch

import sys
sys.path.append('./submodules/gaussian-splatting/')

from random import randint
from scene import Scene, GaussianModel
from gaussian_renderer import render
from source.data_utils import scene_cameras_train_test_split

class Warper3DGS(torch.nn.Module):
    def __init__(self, sh_degree,  opt, pipe, dataset, viewpoint_stack, verbose,
                 do_train_test_split=True):
        super(Warper3DGS, self).__init__()
        """
        Init Warper using all the objects necessary for rendering gaussian splats.
        Here we merely link class objects to the objects instantiated outsided the class.
        """
        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.tmp_radii = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
        self.render = render
        self.gs_config_opt = opt
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.pipe = pipe
        self.scene = Scene(dataset, self.gaussians, shuffle=False)
        if do_train_test_split:
            scene_cameras_train_test_split(self.scene, verbose=verbose)

        self.gaussians.training_setup(opt)
        self.viewpoint_stack = viewpoint_stack
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()

    def forward(self, viewpoint_cam=None):
        """
        For a provided camera viewpoint_cam we render gaussians from this viewpoint.
        If no camera provided then we use the self.viewpoint_stack (list of cameras).
        If the latter is empty we reinitialize it using the self.scene object.
        """
        if not viewpoint_cam:
            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = self.viewpoint_stack[randint(0, len(self.viewpoint_stack) - 1)]

        render_pkg = self.render(viewpoint_cam, self.gaussians, self.pipe, self.bg)
        return render_pkg

