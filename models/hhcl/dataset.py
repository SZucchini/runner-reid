import glob
import os.path as osp
from logging import getLogger, StreamHandler, DEBUG, Formatter

from natsort import natsorted

handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger = getLogger("Log")
logger.setLevel(DEBUG)
for h in logger.handlers[:]:
    logger.removeHandler(h)
    h.close()
logger.addHandler(handler)


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, _ = self.get_imagedata_info(gallery)

        logger.debug("Dataset statistics:")
        logger.debug("  ----------------------------------------")
        logger.debug("  subset   | # ids | # images | # cameras")
        logger.debug("  ----------------------------------------")
        logger.debug("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logger.debug("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logger.debug("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        logger.debug("  ----------------------------------------")


class RunnerCustom(BaseImageDataset):
    def __init__(self, root, verbose=True, **kwargs):
        super(RunnerCustom, self).__init__()
        self.dataset_dir = root
        self._check_before_run()

        train = self._process_dir(self.dataset_dir, relabel=True)
        query = self._process_dir(self.dataset_dir, relabel=False)
        gallery = self._process_dir(self.dataset_dir, relabel=False)

        if verbose:
            logger.debug("=> Custom datasest loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

    def _check_before_run(self):
        if isinstance(self.dataset_dir, list):
            for dirp in self.dataset_dir:
                if not osp.exists(dirp):
                    raise RuntimeError("'{}' is not available".format(dirp))
        else:
            if not osp.exists(self.dataset_dir):
                raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, dir_path, relabel=False):
        if isinstance(dir_path, list):
            imgs_paths = []
            for dirp in dir_path:
                dirs = natsorted(glob.glob(dirp + "/*"))
                for d in dirs:
                    imgs_path = natsorted(glob.glob(d + "/*.jpg"))
                    imgs_paths.append(imgs_path)
        else:
            imgs_paths = []
            dirs = natsorted(glob.glob(dir_path + "/*"))
            for d in dirs:
                imgs_path = natsorted(glob.glob(d + "/*.jpg"))
                imgs_paths.append(imgs_path)

        pid_container = set()
        for imgs_path in imgs_paths:
            pid = int(imgs_path[0].split('/')[-1].split('.')[0])
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for imgs_path in imgs_paths:
            pid = int(imgs_path[0].split('/')[-1].split('.')[0])
            camid = 0
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]
            dataset.append((imgs_path, pid, camid))

        return dataset
