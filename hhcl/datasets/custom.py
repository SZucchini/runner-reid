from __future__ import print_function, absolute_import
import os.path as osp
import glob

from ..utils.data import BaseImageDataset


class Custom(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(Custom, self).__init__()
        self.dataset_dir = root
        self._check_before_run()

        train = self._process_dir(self.dataset_dir, relabel=True)
        query = self._process_dir(self.dataset_dir, relabel=False)
        gallery = self._process_dir(self.dataset_dir, relabel=False)

        if verbose:
            print("=> Custom datasest loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            if "_" in img_path:
                pid = int(img_path.split('/')[-1].split('_')[0])
            else:
                pid = int(img_path.split('/')[-1].split('.')[0])
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            if "_" in img_path:
                pid = int(img_path.split('/')[-1].split('_')[0])
            else:
                pid = int(img_path.split('/')[-1].split('.')[0])
            camid = 0
            if pid == -1:
                continue  # junk images are just ignored
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
