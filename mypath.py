class Path(object):
    @staticmethod
    def dataset_root_dir(dataset):
        if dataset == 'cityscapes' or dataset =="cityscapes_2class":
            return '/media/yazhou/data_drive3/01_dataset/cityscapes/'     # folder that contains leftImg8bit/
        if dataset == 'LaF':
            return '/media/yazhou/data_drive3/01_dataset/LostAndFound/'     # folder that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
