class Config:
    def __init__(self, task):
        if task == "synapse":
            # 修改为正确的路径，例如
            self.base_dir = '/py_skcdf/a18712369256722432680358/mnt/data/SKCDF_extracted/SKCDF/code/data/synapse_splits'
            self.save_dir = '/py_skcdf/a18712369256722432680358/mnt/data/SKCDF_extracted/SKCDF/code/data/synapse_splits'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 1500
        else: # amos
            self.base_dir = '/py_skcdf/a18712369256722432680358/mnt/data/SKCDF_extracted/SKCDF/code/data/amos_splits'
            self.save_dir = '/py_skcdf/a18712369256722432680358/mnt/data/SKCDF_extracted/SKCDF/code/data/amos_splits'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 1500
