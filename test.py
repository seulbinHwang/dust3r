from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode, BasePCOptimizer, PairViewer
from typing import List, Union, Dict, Any, Tuple
import torch

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # 필요한 경우 로컬 체크포인트 경로를 model_name에 지정할 수 있습니다.
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images는 이미지 목록 또는 디렉토리를 받을 수 있습니다.
    images: List[Dict[str, Any]] = load_images([
        'data/left_frames/0.png',
        'data/right_frames/0.png'
    ],
                         size=512)
    # pairs: 길이 2 짜리 리스트. Tuple(1-2쌍), Tuple(2-1쌍)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    pairs = make_pairs(images,
                       scene_graph='complete',
                       prefilter=None,
                       symmetrize=True)
    output: dict = inference(pairs, model, device, batch_size=batch_size)
    """
    view1 (str): Dict
        img 
            tensor (2, 3, 288, 512)
                288, 512: 이미지의 높이와 너비 ???
        true_shape
            tensor (2, 2)
        idx
            list: [1, 0]배치 내에서 이미지의 인덱스를 나타내는 리스트
        instance
            list: ['1', '0']
    view2 (str): Dict
        img 
            tensor (2, 3, 288, 512)
        true_shape
            tensor (2, 2)
        idx
            list: [0, 1]
        instance
            list: ['0', '1']
    pred1 (str): Dict
        pts3d
            tensor: (2, 288, 512, 3)
        conf
            tensor: (2, 288, 512)
    pred2 (str): Dict
        pts3d_in_other_view
            tensor: (2, 288, 512, 3)
        conf
            tensor: (2, 288, 512)
    loss (str): None
    
    """
    # 이 단계에서, raw dust3r 예측을 가지고 있습니다.
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # 여기서 view1, pred1, view2, pred2는 길이가 2인 리스트의 딕셔너리입니다.
    #  -> 대칭화를 하기 때문에 (im1, im2)와 (im2, im1) 쌍을 가지고 있습니다.
    # 각 view에는 다음이 포함됩니다:
    # 이미지 크기: view1['true_shape']와 view2['true_shape']
    # pred1과 pred2는 신뢰도 값을 포함합니다:
    # pred1['conf']와 pred2['conf']
    # pred1은 view1['img'] 공간에서 view1['img']의 3D 포인트를 포함합니다:
    # pred1['pts3d']
    # pred2는 view2['img'] 공간에서 view1['img']의 3D 포인트를 포함합니다:
    # pred2['pts3d_in_other_view']

    # 다음으로 global_aligner를 사용하여 예측을 정렬합니다.
    # 작업에 따라 raw 출력을 그대로 사용해도 될 수 있습니다.
    # 입력 이미지가 두 개뿐인 경우, GlobalAlignerMode.PairViewer를 사용할 수 있습니다:
    #   출력만 변환됩니다.
    #   GlobalAlignerMode.PairViewer를 사용하는 경우,
    #       compute_global_alignment를 실행할 필요가 없습니다.
    scene: PairViewer = global_aligner(output,
                           device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer)#GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst",
                                          niter=niter,
                                          schedule=schedule,
                                          lr=lr)
    """
type(imgs): <class 'list'>
    - len(imgs): 2
    - type(imgs[0]): <class 'numpy.ndarray'>
type(focals): <class 'torch.nn.parameter.Parameter'> # (2)
type(poses): <class 'torch.nn.parameter.Parameter'> # (2, 4, 4)
type(pts3d): <class 'list'>
    - len(pts3d): 2
    - type(pts3d[0]): <class 'torch.Tensor'>
type(confidence_masks): <class 'list'>
    - len(confidence_masks): 2
    - type(confidence_masks[0]): <class 'torch.Tensor'>



    """
    # scene에서 유용한 값을 가져옵니다:
    imgs: List = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()


    # 재구성 시각화
    scene.show()
    raise NotImplementedError("이 코드는 아직 완전히 구현되지 않았습니다.")
    # 두 이미지 간의 2D-2D 매칭 찾기
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(
            *imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(
        *pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # 몇 가지 매칭을 시각화합니다.
    import numpy as np
    from matplotlib import pyplot as pl
    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1,
                                            n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[
        match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                  'constant',
                  constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                  'constant',
                  constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1],
                '-+',
                color=cmap(i / (n_viz - 1)),
                scalex=False,
                scaley=False)
    pl.show(block=True)
