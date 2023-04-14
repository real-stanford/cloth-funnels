import h5py
from filelock import FileLock
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
import time

class Memory:
    log = False
    base_keys = [
        'observations',
        'actions',
        'rewards',
        'is_terminal',
    ]

    def __init__(self, memory_fields=[]):
        self.data = {}
        for key in Memory.base_keys:
            self.data[key] = []
        for memory_field in memory_fields:
            self.data[memory_field] = []

    @staticmethod
    def concat(memories):
        output = Memory()
        for memory in memories:
            for key in memory.data:
                if key not in output.data:
                    output.data[key] = []
                output.data[key].extend(memory.data[key])
        return output

    def clear(self):
        for key in self.data:
            del self.data[key][:]

    def print_length(self):
        output = "[Memory] "
        for key in self.data:
            output += f" {key}: {len(self.data[key])} |"
        print(output)

    def assert_length(self):
        key_lens = [len(self.data[key]) for key in self.data]

        same_length = key_lens.count(key_lens[0]) == len(key_lens)
        if not same_length:
            self.print_length()

    def __len__(self):
        return len(self.data['observations'])

    def add_rewards_and_termination(self, reward, termination):
        if(not( len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions']) - 1\
            == len(self.data['observations']) - 1)):
            print("Error: reward and termination not added correctly")
            print(len(self.data['rewards']), len(self.data['is_terminal']), len(self.data['actions']), len(self.data['observations']))
            raise ValueError(f"Inconsistent data length {len(self.data['rewards'])} {len(self.data['is_terminal'])} {len(self.data['actions'])} {len(self.data['observations'])}")
        self.data['rewards'].append(float(reward))
        self.data['is_terminal'].append(float(termination))

    def add_observation(self, observation):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions'])\
            == len(self.data['observations'])
        self.data['observations'].append(observation)

    def add_action(self, action):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions'])\
            == len(self.data['observations']) - 1
        self.data['actions'].append(action)

    def add_value(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def keys(self):
        return [key for key in self.data]

    def count(self):
        return len(self.data['observations'])

    def done(self):
        if len(self.data['is_terminal']) == 0:
            return False
        return self.data['is_terminal'][-1]

    def get_data(self):
        return self.data

    def check_error(self):
        try:
            count = len(self)
            assert len(self.data['max_coverage']) == count
            assert len(self.data['preaction_coverage']) == count
            assert len(self.data['postaction_coverage']) == count
            return True
        except:
            return False

    def dump(self, hdf5_path, log=False):
        log = True
        print("[Memory] Dumping memory to ", hdf5_path)
        while True:
            try:
                with FileLock(hdf5_path + ".lock"):
                    with h5py.File(hdf5_path, 'a') as file:
                        last_key = None
                        for last_key in file:
                            pass
                        key_idx = int(last_key.split('_')[0])\
                            if last_key is not None else 0
                        while True:
                            group_key = f'{key_idx:09d}'
                            if (group_key + '_step00') not in file\
                                    and (group_key + '_step00_last') not in file:
                                break
                            key_idx += 1
                        for step in range(len(self)):
                            step_key = group_key + f'_step{step:02d}'
                            if step == len(self) - 1:
                                step_key += '_last'
                            try:
                                group = file.create_group(step_key)
                            except Exception as e:
                                print(e, step_key)
                                group = file.create_group(
                                    step_key + '_' +
                                    str(random.randint(0, int(1e5))))
                            for key, value in self.data.items():
                                try:
                                    if key in ['cloth_mass', 'task_name', 'cloth_instance', 'adaptive_scale']:
                                        continue
                                    if any(key == skip_key
                                        for skip_key in
                                        ['visualization_dir', 'faces',
                                            'gripper_states', 'states', 'cloth_mass', 'task_name']) \
                                            and step != 0:
                                        continue
                                    step_value = value[step]
                                    if type(step_value) == float\
                                        or type(step_value) == np.float64\
                                        or type(step_value) == str\
                                        or type(step_value) == int:
                                        if type(step_value) == float\
                                            or type(step_value) == np.float64:
                                            if np.isnan(step_value) or np.isinf(step_value):
                                                raise ValueError(f"NaN or Inf value in {key}")

                                        group.attrs[key] = step_value
                                    elif type(step_value) == list:
                                        subgroup = group.create_group(key)
                                        for i, item in enumerate(step_value):
                                            subgroup.create_dataset(
                                                name=f'{i:09d}',
                                                data=item,
                                                compression='gzip')
                                    else:
                                        group.create_dataset(
                                            name=key,
                                            data=step_value,
                                            compression='gzip')
                                except Exception as e:
                                    print(f'[Memory] Dump key {key} error:', e)
                                    print(value)
                        return group_key
            except Exception as e:
                print("[Memory] Dump error:", e)
                time.sleep(0.1)
                pass