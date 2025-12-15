def scene_cameras_train_test_split(scene, verbose=False):
    """
    Iterate over resolutions in the scene. For each resolution check if this resolution has test_cameras
    if it doesn't then extract every 8th camera from the train and put it to the test set. This follows the
    evaluation protocol suggested by Kerbl et al. in the seminal work on 3DGS. All changes to the input
    object scene are inplace changes.
    :param scene: Scene Class object from the gaussian-splatting.scene module
    :param verbose: Print initial and final stage of the function
    :return:  None

    """
    if verbose: print("Preparing train and test sets split...")
    for resolution in scene.train_cameras.keys():
        if len(scene.test_cameras[resolution]) == 0:
            if verbose:
                print(f"Found no test_cameras for resolution {resolution}. Move every 8th camera out ouf total "+\
                      f"{len(scene.train_cameras[resolution])} train cameras to the test set now")
            N = len(scene.train_cameras[resolution])
            scene.test_cameras[resolution] = [scene.train_cameras[resolution][idx] for idx in range(0, N) 
                                              if idx % 8 == 0]
            scene.train_cameras[resolution] = [scene.train_cameras[resolution][idx] for idx in range(0, N)
                                               if idx % 8 != 0]
            if verbose:
                print(f"Done. Now train and test sets contain each {len(scene.train_cameras[resolution])} and " + \
                      f"{len(scene.test_cameras[resolution])} cameras respectively.")


    return
