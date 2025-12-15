import torch
from random import randint
from tqdm.rich import trange
from tqdm import tqdm as tqdm
from source.networks import Warper3DGS
#import wandb
import sys
import os
import numpy as np

sys.path.append('./submodules/gaussian-splatting/')
import lpips
from source.losses import ssim, l1_loss, psnr
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

from source.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from source.utils_aux import log_samples

from source.timer import Timer

class EDGSTrainer:
    def __init__(self,
                 GS: Warper3DGS,
                 training_config,
                 dataset_white_background=False,
                 device=torch.device('cuda'),
                 log_wandb=False,
                 ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.gaussians = GS.gaussians

        self.training_config = training_config
        self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = training_config.save_iterations
        self.evaluate_iterations = None
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        # Logs in the format {step:{"loss1":loss1_value, "loss2":loss2_value}}
        self.logs_losses = {}
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.device = device
        self.timer = Timer()
        self.log_wandb = log_wandb

    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            self.gs.gaussians.restore(
                torch.load(f"{load_cfg.gs}/chkpnt{load_cfg.gs_step}.pth")[0],
                self.training_config)
            self.GS_optimizer = self.GS.gaussians.optimizer
            self.CONSOLE.print(f"3DGS loaded from checkpoint for iteration {load_cfg.gs_step}",
                               style="info")
            self.training_step += load_cfg.gs_step
            self.gs_step += load_cfg.gs_step

    def train(self, train_cfg):
        # 3DGS training
        self.CONSOLE.print("Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info")    
        with trange(self.training_step, self.training_step + train_cfg.gs_epochs, desc="[green]Train gaussians") as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify)
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if self.gs_step < self.training_config.densify_until_iter and self.gs_step % 10 == 0:
                            opacities_new = torch.log(torch.exp(self.GS.gaussians._opacity.data) * 0.99)
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix({"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"}, refresh=True)
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (self.training_step <= 3000 and self.training_step % 500 == 0) or \
                            (self.training_step > 3000 and self.training_step % 1000 == 228) :
                            self.evaluate()

                    self.timer.start()


    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras(), 'cam_idx': self.training_config.TEST_CAM_IDX_TO_LOG},
                              {'name': 'train',
                               'cameras': [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in
                                           range(0, 150, 5)], 'cam_idx': 10})
        '''if self.log_wandb:
            wandb.log({f"Number of Gaussians": len(self.GS.gaussians._xyz)}, step=self.training_step)'''
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(self.GS(viewpoint)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += self.lpips(image, gt_image).detach().double()
                    if idx in [config['cam_idx']]:
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_splat_test /= len(config['cameras'])
                '''if self.log_wandb:
                    wandb.log({f"{config['name']}/L1": l1_test.item(), f"{config['name']}/PSNR": psnr_test.item(), \
                            f"{config['name']}/SSIM": ssim_test.item(), f"{config['name']}/LPIPS_splat": lpips_splat_test.item()}, step = self.training_step)'''
                self.CONSOLE.print("\n[ITER {}], #{} gaussians, Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                    self.training_step, len(self.GS.gaussians._xyz), config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_splat_test.item()), style="info")
        if self.log_wandb:
            with torch.no_grad():
                log_samples(torch.stack((log_real_images[0],log_gen_images[0])) , [], self.training_step, caption="Real and Generated Samples")
                #wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr = False, no_densify = False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
      
        render_pkg = self.GS(viewpoint_cam=viewpoint_cam)
        image = render_pkg["render"]
        # Loss
        gt_image = viewpoint_cam.original_image.to(self.device)
        L1_loss = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - self.training_config.lambda_dssim) * L1_loss + \
               self.training_config.lambda_dssim * ssim_loss
        self.timer.pause() 
        self.logs_losses[self.training_step] = {"loss": loss.item(),
                                                "L1_loss": L1_loss.item(),
                                                "ssim_loss": ssim_loss.item()}
        
        '''if self.log_wandb:
            for k, v in self.logs_losses[self.training_step].items():
                wandb.log({f"train/{k}": v}, step=self.training_step)'''
        self.ema_loss_for_log = 0.4 * self.logs_losses[self.training_step]["loss"] + 0.6 * self.ema_loss_for_log
        self.timer.start()
        self.GS_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if self.gs_step < self.training_config.densify_until_iter and not no_densify:
                self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]])
                self.GS.gaussians.add_densification_stats(render_pkg["viewspace_points"],
                                                                     render_pkg["visibility_filter"])

        # Optimizer step
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True)
        return render_pkg["radii"]

    def densify_and_prune(self, radii = None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if (self.gs_step > self.training_config.densify_from_iter) and \
                    (self.gs_step % self.training_config.densification_interval == 0):
                size_threshold = 20 if self.gs_step > self.training_config.opacity_reset_interval else None
                try:
                    self.GS.gaussians.densify_and_prune(
                        self.training_config.densify_grad_threshold,
                        0.005,
                        self.GS.scene.cameras_extent,
                        radii,
                        200000
                    )
                except TypeError:
                    try:
                        self.GS.gaussians.densify_and_prune(
                self.training_config.densify_grad_threshold,
                            0.005,
                            self.GS.scene.cameras_extent,
                            radii
                        )
                    except TypeError:
                        self.GS.gaussians.densify_and_prune(
                            self.training_config.densify_grad_threshold,
                            0.005,
                            self.GS.scene.cameras_extent
                        )
                '''self.GS.gaussians.densify_and_prune(self.training_config.densify_grad_threshold,
                                                               0.005,
                                                               self.GS.scene.cameras_extent,
                                                               size_threshold,radii)'''
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                    self.dataset_white_background and self.gs_step == self.training_config.densify_from_iter):
                self.GS.gaussians.reset_opacity()             

          

    def save_model(self):
        print("\n[ITER {}] Saving Gaussians".format(self.gs_step))
        self.scene.save(self.gs_step)
        print("\n[ITER {}] Saving Checkpoint".format(self.gs_step))
        torch.save((self.GS.gaussians.capture(), self.gs_step),
                self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")


    def init_with_corr(self, cfg, verbose=False, roma_model=None): 
        """
        Initializes image with matchings. Also removes SfM init points.
        Args:
            cfg: configuration part named init_wC. Check train.yaml
            verbose: whether you want to print intermediate results. Useful for debug.
            roma_model: optionally you can pass here preinit RoMA model to avoid reinit 
                it every time.  
        """
        if not cfg.use:
            return None
        N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", N_splats_at_init)
        if cfg.nns_per_ref == 1:
            init_fn = init_gaussians_with_corr_fast
        else:
            init_fn = init_gaussians_with_corr
        camera_set, selected_indices, visualization_dict = init_fn(
            self.GS.gaussians, 
            self.scene, 
            cfg, 
            self.device,                                                                                    
            verbose=verbose,
            roma_model=roma_model)
        
        # === ADDED: compute baseline metrics and persist seed metadata before pruning ===
        #only for seed meta
        '''
        try:
            seed_start = int(visualization_dict.get("seed_new_gauss_start", -1))
            seed_count = int(visualization_dict.get("seed_new_gauss_count", 0))

            if seed_start >= 0 and seed_count > 0:
                # Raw meta direct from corr_init
                ref_idx = visualization_dict["seed_ref_cam_idx"]        # (N,)
                nbr_idx = visualization_dict["seed_nbr_cam_idx"]        # (N,)
                uv_ref  = visualization_dict["seed_uv_ref_norm"]        # (N,2)
                uv_nbr  = visualization_dict["seed_uv_nbr_norm"]        # (N,2)
                e_mean  = visualization_dict["seed_reproj_err_mean"]    # (N,)
                e_max   = visualization_dict["seed_reproj_err_max"]     # (N,)
                m_conf  = visualization_dict["seed_match_conf"]         # (N,)

                # Slice 3D positions of newly inserted gaussians
                xyz_new = self.GS.gaussians._xyz[seed_start: seed_start + seed_count].detach().cpu().numpy()  # (N,3)

                # Compute baseline angle & baseline/depth
                C_ref = np.stack([camera_set[int(i)].camera_center.detach().cpu().numpy() for i in ref_idx], axis=0)  # (N,3)
                C_nbr = np.stack([camera_set[int(i)].camera_center.detach().cpu().numpy() for i in nbr_idx], axis=0)  # (N,3)
                v1 = C_ref - xyz_new
                v2 = C_nbr - xyz_new

                # angles in degrees
                dot = np.einsum('ij,ij->i', v1, v2)
                n1  = np.linalg.norm(v1, axis=1) + 1e-8
                n2  = np.linalg.norm(v2, axis=1) + 1e-8
                cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
                baseline_angle_deg = np.degrees(np.arccos(cos)).astype(np.float32)

                # baseline-to-depth
                baseline_len = np.linalg.norm(C_ref - C_nbr, axis=1)
                depth_avg    = 0.5 * (n1 + n2) + 1e-8
                baseline_to_depth = (baseline_len / depth_avg).astype(np.float32)

                # Save to disk
                out_dir = os.path.join(self.scene.model_path, "edgs_meta")
                os.makedirs(out_dir, exist_ok=True)

                seed_xyz_np = visualization_dict.get("seed_xyz", None)

                np.savez_compressed(
                    os.path.join(out_dir, "seed_meta.npz"),
                    ref_cam_idx=ref_idx.astype(np.int32),
                    nbr_cam_idx=nbr_idx.astype(np.int32),
                    uv_ref_norm=uv_ref.astype(np.float32),
                    uv_nbr_norm=uv_nbr.astype(np.float32),
                    reproj_err_mean=e_mean.astype(np.float32),
                    reproj_err_max=e_max.astype(np.float32),
                    match_conf=m_conf.astype(np.float32),
                    new_gauss_start=np.int64(seed_start),
                    new_gauss_count=np.int64(seed_count),
                    seed_xyz=seed_xyz_np.astype(np.float32) if seed_xyz_np is not None else np.zeros((0,3), dtype=np.float32),
                )
                np.savez_compressed(
                    os.path.join(out_dir, "seed_geom_meta.npz"),
                    baseline_angle_deg=baseline_angle_deg,
                    baseline_to_depth=baseline_to_depth,
                )
                print(f"[EDGS] Saved seed metadata to {out_dir}")

        except Exception as ex:
            print(f"[EDGS] Warning: failed to persist seed metadata: {ex}")'''


        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(self.gaussians._xyz.shape[0]).to(self.device)
                mask = torch.concat([torch.ones(N_splats_at_init, dtype=torch.bool),
                                    torch.zeros(N_splats_after_init-N_splats_at_init, dtype=torch.bool)],
                                axis=0)
                self.GS.gaussians.prune_points(mask)
        with torch.no_grad():
            gaussians =  self.gaussians
            gaussians._scaling =  gaussians.scaling_inverse_activation(gaussians.scaling_activation(gaussians._scaling)*0.5)
        return visualization_dict
    

    def prune(self, radii, min_opacity=0.005):
        self.GS.gaussians.tmp_radii = radii
        if self.gs_step < self.training_config.densify_until_iter:
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            self.GS.gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
        self.GS.gaussians.tmp_radii = None

