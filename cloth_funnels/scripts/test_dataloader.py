
from cloth_funnels.utils.utils import get_loader
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import tqdm
import h5py

dataset_path = '/local/crv/acanberk/cloth-funnels/experiments/04-02-1312-longsleeve/replay_buffer.hdf5'

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(args : DictConfig):

    #override args.num_workers
    for i in range(0, 32, 2):
        print("=====================================")
        print("Testing with {} workers".format(i))
        args.num_workers = i
        
        with h5py.File(dataset_path, 'r') as f:
            loaders = (get_loader(
                    hdf5_path=dataset_path,
                    filter_fn=lambda attrs:
                        (attrs['action_primitive'] == action_primitive \
                            and ('preaction_l2_distance' in attrs)), **args) for action_primitive in args.action_primitives)
            
        start = time.time()
        for loader in loaders:
            for i, data in enumerate(loader):
                pass
        end = time.time()
        print("Time taken: {}".format(end - start))
        print("=====================================")

        




if __name__ == "__main__":
    main()
