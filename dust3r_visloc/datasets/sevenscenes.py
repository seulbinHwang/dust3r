# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# 7 Scenes dataloader
# --------------------------------------------------------
import os
import numpy as np
import torch
import PIL.Image

import kapture
from kapture.io.csv import kapture_from_dir
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file
from kapture.io.records import depth_map_from_file

from dust3r_visloc.datasets.utils import cam_to_world_from_kapture, get_resize_function, rescale_points3d
from dust3r_visloc.datasets.base_dataset import BaseVislocDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, xy_grid, geotrf

"""
위 코드는 "7-scenes 데이터셋을 로드하고 처리"하는 데이터 로더 클래스인 `VislocSevenScenes`
이 클래스는 시각적 위치 추정 작업에 필요한 데이터를 준비하고, 
    데이터셋에서 쿼리 이미지와 매핑 이미지를 로드하여 필요한 형식으로 변환
주요 역할과 기능은 다음과 같습니다:

### 주요 클래스 및 함수

1. **VislocSevenScenes 클래스**:
   - `BaseVislocDataset` 클래스를 상속받아 7-scenes 데이터셋을 처리
   - 초기화 시 루트 경로, 서브씬, 이미지 쌍 파일 등을 입력으로 받습니다.

2. **__init__ 메서드**:
   - 루트 디렉토리, 서브씬, 쌍 파일 등을 초기화
   - 쿼리와 매핑 데이터를 각각 로드하여 kapture 형식으로 변환
   - `get_ordered_pairs_from_file` 함수를 사용하여 쿼리 이미지와 매핑 이미지 쌍을 로드

3. **__len__ 메서드**:
   - 데이터셋의 길이를 반환합니다. 이는 쿼리 이미지의 수를 의미합니다.

4. **__getitem__ 메서드**:
   - 주어진 인덱스에 해당하는 쿼리 이미지와 매핑 이미지를 로드하고, 필요한 전처리를 수행합니다.
   - 이미지와 관련된 메타데이터(내부 파라미터, 외부 파라미터 등)를 로드합니다.
   - 이미지를 리사이즈하고, 정규화된 텐서로 변환합니다.
   - 필요할 경우 깊이 맵을 로드하고, 이를 절대 카메라 좌표로 변환합니다.
   - 3D 포인트와 2D 포인트를 계산하고, 이를 텐서 형식으로 변환합니다.
   - 뷰 정보를 딕셔너리 형태로 정리하여 반환합니다.

### 데이터 처리 과정

1. **쿼리 및 매핑 데이터 로드**:
   - `kapture_from_dir` 함수를 사용하여 쿼리와 매핑 데이터를 로드합니다.
   - `searchindex`를 생성하여 이미지 이름을 타임스탬프와 센서 ID로 매핑합니다.

2. **이미지 쌍 파일 로드**:
   - `get_ordered_pairs_from_file` 함수를 사용하여 쿼리 이미지와 매핑 이미지의 쌍을 로드합니다.

3. **이미지 및 메타데이터 처리**:
   - 각 이미지에 대해 내부 파라미터, 외부 파라미터, 왜곡 계수 등을 로드합니다.
   - 이미지를 리사이즈하고 정규화된 텐서로 변환합니다.
   - 필요할 경우 깊이 맵을 로드하여 절대 카메라 좌표로 변환하고, 3D 포인트와 2D 포인트를 계산합니다.

4. **결과 반환**:
   - 쿼리 이미지와 매핑 이미지의 뷰 정보를 딕셔너리 형태로 정리하여 리스트로 반환합니다.

"""
class VislocSevenScenes(BaseVislocDataset):

    def __init__(self, root, subscene, pairsfile, topk=1):
        super().__init__()
        self.root = root
        self.subscene = subscene
        self.topk = topk
        self.num_views = self.topk + 1
        self.maxdim = None
        self.patch_size = None

        query_path = os.path.join(self.root, subscene, 'query')
        kdata_query = kapture_from_dir(query_path)
        assert kdata_query.records_camera is not None and kdata_query.trajectories is not None and kdata_query.rigs is not None
        kapture.rigs_remove_inplace(kdata_query.trajectories, kdata_query.rigs)
        kdata_query_searchindex = {
            kdata_query.records_camera[(timestamp, sensor_id)]:
                (timestamp, sensor_id)
            for timestamp, sensor_id in kdata_query.records_camera.key_pairs()
        }
        self.query_data = {
            'path': query_path,
            'kdata': kdata_query,
            'searchindex': kdata_query_searchindex
        }

        map_path = os.path.join(self.root, subscene, 'mapping')
        kdata_map = kapture_from_dir(map_path)
        assert kdata_map.records_camera is not None and kdata_map.trajectories is not None and kdata_map.rigs is not None
        kapture.rigs_remove_inplace(kdata_map.trajectories, kdata_map.rigs)
        kdata_map_searchindex = {
            kdata_map.records_camera[(timestamp, sensor_id)]:
                (timestamp, sensor_id)
            for timestamp, sensor_id in kdata_map.records_camera.key_pairs()
        }
        self.map_data = {
            'path': map_path,
            'kdata': kdata_map,
            'searchindex': kdata_map_searchindex
        }

        self.pairs = get_ordered_pairs_from_file(
            os.path.join(self.root, subscene, 'pairfiles/query',
                         pairsfile + '.txt'))
        self.scenes = kdata_query.records_camera.data_list()

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        assert self.maxdim is not None and self.patch_size is not None
        query_image = self.scenes[idx]
        map_images = [p[0] for p in self.pairs[query_image][:self.topk]]
        views = []
        dataarray = [(query_image, self.query_data, False)] + [
            (map_image, self.map_data, True) for map_image in map_images
        ]
        for idx, (imgname, data, should_load_depth) in enumerate(dataarray):
            imgpath, kdata, searchindex = map(data.get,
                                              ['path', 'kdata', 'searchindex'])

            timestamp, camera_id = searchindex[imgname]

            # for 7scenes, SIMPLE_PINHOLE
            camera_params = kdata.sensors[camera_id].camera_params
            W, H, f, cx, cy = camera_params
            distortion = [0, 0, 0, 0]
            intrinsics = np.float32([(f, 0, cx), (0, f, cy), (0, 0, 1)])

            cam_to_world = cam_to_world_from_kapture(kdata, timestamp,
                                                     camera_id)

            # Load RGB image
            rgb_image = PIL.Image.open(
                os.path.join(imgpath, 'sensors/records_data',
                             imgname)).convert('RGB')
            rgb_image.load()

            W, H = rgb_image.size
            resize_func, to_resize, to_orig = get_resize_function(
                self.maxdim, self.patch_size, H, W)

            rgb_tensor = resize_func(ImgNorm(rgb_image))

            view = {
                'intrinsics': intrinsics,
                'distortion': distortion,
                'cam_to_world': cam_to_world,
                'rgb': rgb_image,
                'rgb_rescaled': rgb_tensor,
                'to_orig': to_orig,
                'idx': idx,
                'image_name': imgname
            }

            # Load depthmap
            if should_load_depth:
                depthmap_filename = os.path.join(
                    imgpath, 'sensors/records_data',
                    imgname.replace('color.png', 'depth.reg'))
                depthmap = depth_map_from_file(
                    depthmap_filename, (int(W), int(H))).astype(np.float32)
                pts3d_full, pts3d_valid = depthmap_to_absolute_camera_coordinates(
                    depthmap, intrinsics, cam_to_world)

                pts3d = pts3d_full[pts3d_valid]
                pts2d_int = xy_grid(W, H)[pts3d_valid]
                pts2d = pts2d_int.astype(np.float64)

                # nan => invalid
                pts3d_full[~pts3d_valid] = np.nan
                pts3d_full = torch.from_numpy(pts3d_full)
                view['pts3d'] = pts3d_full
                view["valid"] = pts3d_full.sum(dim=-1).isfinite()

                HR, WR = rgb_tensor.shape[1:]
                _, _, pts3d_rescaled, valid_rescaled = rescale_points3d(
                    pts2d, pts3d, to_resize, HR, WR)
                pts3d_rescaled = torch.from_numpy(pts3d_rescaled)
                valid_rescaled = torch.from_numpy(valid_rescaled)
                view['pts3d_rescaled'] = pts3d_rescaled
                view["valid_rescaled"] = valid_rescaled
            views.append(view)
        return views
