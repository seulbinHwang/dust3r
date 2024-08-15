# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dummy optimizer for visualizing pairs
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.spatial.transform import Rotation as R_scipy

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import inv, geotrf, depthmap_to_absolute_camera_coordinates
from dust3r.cloud_opt.commons import edge_str
from dust3r.post_process import estimate_focal_knowing_depth
"""
### `PairViewer` 클래스의 목적과 내용

#### 목적

`PairViewer` 클래스는 두 이미지 쌍에 대한 "시각적 결과를 시각화"하기 위한 더미 최적화 도구
이 클래스는 주로 결과를 확인하고 디버깅하는 데 사용
이를 통해 이미지 쌍 간의 관계를 시각화하고, 각 이미지의 3D 포인트 클라우드 및 포즈를 계산할 수 있음

#### 요약

- 두 이미지 간의 3D 포인트 클라우드 및 포즈를 계산하여 시각화하고, 결과를 디버깅하는 데 사용
- 각 이미지의 3D 포인트 클라우드를 계산하고, 이미지 간의 상대적인 포즈를 추정합니다.
- 이미지 쌍 간의 신뢰도를 계산하고, 신뢰도가 높은 쪽의 포즈를 기준으로 데이터를 설정합니다.
- 이 클래스는 실제 학습 또는 예측보다는 결과를 시각적으로 확인하고 디버깅하는 데 주로 사용됩니다.


#### 주요 기능 및 내용

1. **초기화 (`__init__` 메서드)**:
   - 입력 파라미터를 받아 초기화를 수행
   - 이미지 쌍이 대칭인지 확인하고, 
        - 각 이미지 쌍에 대한 신뢰도(confidence)와 초점 거리(focal length)를 계산
   - 각 이미지에 대한 3D 포인트 클라우드를 추정하고, 
        - 이를 바탕으로 이미지 간의 상대적인 포즈를 계산
   - 신뢰도가 높은 쪽의 포즈를 기준으로 
        - 포인트 클라우드(depth) 및 이미지 포즈(im_poses)를 설정

2. **_set_depthmap 메서드**:
   - 깊이 맵을 설정하는 메서드지만, `PairViewer`에서는 무시됩니다. 
   - 이는 디버깅 또는 시각화 목적이기 때문에 깊이 맵 설정을 필요로 하지 않습니다.

3. **get_depthmaps 메서드**:
   - 각 이미지의 깊이 맵을 반환합니다.

4. **_set_focal 메서드**:
   - 주어진 인덱스의 초점 거리를 설정합니다.

5. **get_focals 메서드**:
   - 초점 거리 목록을 반환

6. **get_known_focal_mask 메서드**:
   - 초점 거리가 고정된(fixed) 여부를 나타내는 마스크를 반환

7. **get_principal_points 메서드**:
   - 주점(principal point) 목록을 반환

8. **get_intrinsics 메서드**:
   - 카메라 내적 행렬(intrinsics) 목록을 반환

9. **get_im_poses 메서드**:
   - 이미지 포즈 목록을 반환

10. **depth_to_pts3d 메서드**:
   - 깊이 맵을 3D 포인트 클라우드로 변환하여 반환

11. **forward 메서드**:
   - 더미 메서드로, 나노(NaN) 값을 반환합니다. 
   - 이 클래스는 주로 시각화 목적이기 때문에 실제로 forward 연산을 수행하지 않습니다.


"""


def rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """
    Roll, pitch, yaw 각도를 사용하여 quaternion (w, x, y, z)를 계산합니다.

    Args:
        roll (float): 롤(Roll) 각도 (라디안 단위)
        pitch (float): 피치(Pitch) 각도 (라디안 단위)
        yaw (float): 요(Yaw) 각도 (라디안 단위)

    Returns:
        np.ndarray: 쿼터니언을 나타내는 (w, x, y, z) 형태의 numpy 배열
    """
    roll, pitch, yaw = rpy
    # Roll, Pitch, Yaw를 Rotation 객체로 변환
    rotation = R_scipy.from_euler('xyz', [roll, pitch, yaw], degrees=False)

    # Rotation 객체를 quaternion (w, x, y, z) 형태로 변환하여 반환
    quaternion = rotation.as_quat()

    return quaternion


def average_transformation(transformations: np.ndarray) -> np.ndarray:
    """
    여러 개의 좌표계 변환 (translation 및 rotation)을 평균하여 최종 평균 변환을 계산합니다.

    Args:
        transformations (np.ndarray): shape이 (n, 7)인 numpy 배열.
                                       각 transformation은 (tx, ty, tz, qw, qx, qy, qz)의 형태입니다.
                                       여기서 (tx, ty, tz)는 translation, (qw, qx, qy, qz)는 rotation을 의미합니다.

    Returns:
        np.ndarray: shape이 (7,)인 numpy 배열로, 평균 translation과 평균 rotation(quaternion)이 결합된 최종 변환입니다.
    """

    # 1. Translation 평균 계산
    translations = transformations[:, :3]  # (n, 3)
    avg_translation = np.mean(translations, axis=0)  # (3,)

    # 2. Rotation 평균 계산
    quaternions = transformations[:, 3:]  # (n, 4)

    # scipy의 Rotation 클래스를 사용하여 Karcher mean (geometric mean)을 계산
    rotations = R_scipy.from_quat(quaternions)  # Rotation 객체로 변환
    avg_rotation = rotations.mean()  # 평균을 계산
    avg_quaternion = avg_rotation.as_quat()  # 평균 쿼터니언을 다시 (w, x, y, z)로 변환

    # 3. 최종 평균 transformation 계산
    avg_transformation = np.hstack((avg_translation, avg_quaternion))  # (7,)

    return avg_transformation


class PairViewer(BasePCOptimizer):
    """
    This a Dummy Optimizer.
    To use only when the goal is to visualize the results for a pair of images (with is_symmetrized)
    """

    def __init__(self, *args, **kwargs):
        """ output
        view1 (str): Dict
            img
                tensor (2, 3, 288, 512)
                    288, 512: 이미지의 높이와 너비 ???
            true_shape
                tensor (2, 2)
            idx
                list: [1, 0] 배치 내에서 이미지의 인덱스를 나타내는 리스트
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
        super().__init__(*args, **kwargs)
        assert self.is_symmetrized and self.n_edges == 2
        self.has_im_poses = True

        # compute all parameters directly from raw input
        self.focals = []
        self.pp = []
        rel_poses = []
        average_confidences_per_pair = []
        hobin_transformations = []
        seulbin_transformations = []
        for i in range(self.n_imgs):  # 2장
            """
            edge_str(0, 1) = '0_1'
            edge_str(1, 0) = '1_0'
            """
            # i=0, conf: (0,1) 쌍에서, 두 이미지가 생성한 pts에 대한 평균 신뢰도 값
            # i=1, conf: (1,0) 쌍에서, 두 이미지가 생성한 pts에 대한 평균 신뢰도 값
            average_confidence = float(self.conf_i[edge_str(i, 1 - i)].mean() *
                                       self.conf_j[edge_str(i, 1 - i)].mean())
            if self.verbose:
                print(f'  - {average_confidence=:.3} for edge {i}-{1-i}')
            average_confidences_per_pair.append(average_confidence)
            # self.imshapes = [(288, 512), (288, 512)]
            H, W = self.imshapes[i]
            """
            edge_str(0, 1) = '0_1'
            edge_str(1, 0) = '1_0'
            
            self.pred_i: ParameterDict
                (1_0): (288, 512, 3) # (1,0) 쌍에서 view1의 pts (view1 좌표계 기준)
                (0_1): (288, 512, 3) # (0,1) 쌍에서 view1의 pts (view 1 좌표계 기준)

            # i=0, main_cam_pts3d: (0,1) 쌍에서 view1(0번)의 pts (view 1(0번) 좌표계 기준) (288, 512, 3)
            # i=1, main_cam_pts3d: (1,0) 쌍에서 view1(1번)의 pts (view 1(1번) 좌표계 기준) (288, 512, 3)
            """
            main_cam_pts3d = self.pred_i[edge_str(i, 1 - i)]  # (288, 512, 3)
            # TODO: 여기의 값을 바꿔야 함. (주점과 초점거리)
            main_cam_pp = torch.tensor((W / 2, H / 2))
            """
            호빈-슬빈
            슬빈-호빈
            """
            main_cam_focal = float(
                estimate_focal_knowing_depth(main_cam_pts3d[None],
                                             main_cam_pp,
                                             focal_mode='weiszfeld'))
            print("main_cam_pp:", main_cam_pp)
            print("main_cam_focal:", main_cam_focal)
            self.focals.append(main_cam_focal)
            self.pp.append(main_cam_pp)

            # estimate the pose of pts1 in image 2
            pixels = np.mgrid[:W, :H].T.astype(np.float32)  # (H, W, 2)
            """
            edge_str(0, 1) = '0_1'
            edge_str(1, 0) = '1_0'
            self.pred_j: ParameterDict
                (1_0): (288, 512, 3) # (1,0) 쌍에서 view2의 pts (view1 좌표계 기준)
                (0_1): (288, 512, 3) # (0,1) 쌍에서 view2의 pts (view1 좌표계 기준)

            # i=0, main_cam_pts3d: (0,1) 쌍에서 view2의(1번)의 pts (view 1(0번) 좌표계 기준) (288, 512, 3)
            # i=1, main_cam_pts3d: (1,0) 쌍에서 view2의(0번)의 pts (view 1(1번) 좌표계 기준) (288, 512, 3)
            """
            other_cam_pts3d = self.pred_j[edge_str(1 - i,
                                                   i)].numpy()  # (288, 512, 3)
            assert other_cam_pts3d.shape[:2] == (H, W)
            # other_cam_pts3d 중, 신뢰도가 self.min_conf_thr 이상인 값만 True로 설정
            other_cam_conf_mask = self.get_masks()[i].numpy()  # (288, 512)
            K = np.float32([(main_cam_focal, 0, main_cam_pp[0]),
                            (0, main_cam_focal, main_cam_pp[1]), (0, 0, 1)])

            try:
                res = cv2.solvePnPRansac(other_cam_pts3d[other_cam_conf_mask],
                                         pixels[other_cam_conf_mask],
                                         K,
                                         None,
                                         iterationsCount=100,
                                         reprojectionError=5,
                                         flags=cv2.SOLVEPNP_SQPNP)
                success, R, T, inliers = res
                if i == 0:
                    print("슬빈 cam이 main 일 때, 호빈 cam의 포즈")
                    print("R:", R)  # (3, 1)
                    print("T:", T) # (3, 1)
                    R_ = R.squeeze()
                    quaternion = rpy_to_quaternion(R_)
                    transformation_ = np.hstack((T.squeeze(), quaternion)) # (7,)
                    hobin_transformations.append(transformation_)
                    raise ValueError
                else:
                    print("호빈 cam이 main 일 때, 슬빈 cam의 포즈")
                    print("R:", R)
                    print("T:", T)
                    R_ = R.squeeze()
                    quaternion = rpy_to_quaternion(R_)
                    transformation_ = np.hstack((T.squeeze(), quaternion))
                    seulbin_transformations.append(transformation_)
                assert success

                R = cv2.Rodrigues(R)[0]  # world to cam
                pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
            except:
                pose = np.eye(4)
            rel_poses.append(torch.from_numpy(pose.astype(np.float32)))
        hobin_transformations = np.array(hobin_transformations)
        seulbin_transformations = np.array(seulbin_transformations) # (2, 7)
        hobin_transformations = average_transformation(hobin_transformations)
        seulbin_transformations = average_transformation(seulbin_transformations)
        # let's use the pair with the most confidence
        # (0,1)쌍의 결과가, (1,0) 쌍의 결과보다 신뢰도가 높으면
        if average_confidences_per_pair[0] > average_confidences_per_pair[1]:
            # ptcloud is expressed in camera1
            self.im_poses = [torch.eye(4), rel_poses[1]]  # I, cam2-to-cam1
            self.depth = [
                self.pred_i['0_1'][..., 2],
                geotrf(inv(rel_poses[1]), self.pred_j['0_1'])[..., 2]
            ]
        else:
            # ptcloud is expressed in camera2
            self.im_poses = [rel_poses[0], torch.eye(4)]  # I, cam1-to-cam2
            self.depth = [
                geotrf(inv(rel_poses[0]), self.pred_j['1_0'])[..., 2],
                self.pred_i['1_0'][..., 2]
            ]

        self.im_poses = nn.Parameter(torch.stack(self.im_poses, dim=0),
                                     requires_grad=False)
        self.focals = nn.Parameter(torch.tensor(self.focals),
                                   requires_grad=False)
        self.pp = nn.Parameter(torch.stack(self.pp, dim=0), requires_grad=False)
        self.depth = nn.ParameterList(self.depth)
        for p in self.parameters():
            p.requires_grad = False

    def _set_depthmap(self, idx, depth, force=False):
        if self.verbose:
            print('_set_depthmap is ignored in PairViewer')
        return

    def get_depthmaps(self, raw=False):
        depth = [d.to(self.device) for d in self.depth]
        return depth

    def _set_focal(self, idx, focal, force=False):
        self.focals[idx] = focal

    def get_focals(self):
        return self.focals

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.focals])

    def get_principal_points(self):
        return self.pp

    def get_intrinsics(self):
        focals = self.get_focals()
        pps = self.get_principal_points()
        K = torch.zeros((len(focals), 3, 3), device=self.device)
        for i in range(len(focals)):
            K[i, 0, 0] = K[i, 1, 1] = focals[i]
            K[i, :2, 2] = pps[i]
            K[i, 2, 2] = 1
        return K

    def get_im_poses(self):
        return self.im_poses

    def depth_to_pts3d(self):
        pts3d = []
        for d, intrinsics, im_pose in zip(self.depth, self.get_intrinsics(),
                                          self.get_im_poses()):
            pts, _ = depthmap_to_absolute_camera_coordinates(
                d.cpu().numpy(),
                intrinsics.cpu().numpy(),
                im_pose.cpu().numpy())
            pts3d.append(torch.from_numpy(pts).to(device=self.device))
        return pts3d

    def forward(self):
        return float('nan')
