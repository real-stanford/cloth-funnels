
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np

def pick_nth_step(steps, n):
    num_steps = len(steps)
    if n >= num_steps:
        return steps.iloc[-1]
    return steps.iloc[n]

def early_stop_pred(steps, n):
    num_steps = len(steps)
    if n >= len(steps):
        return steps.iloc[-1]
    for step in range(n, num_steps):
        if steps.iloc[step].predicted_weighted_value < 0:
            return steps.iloc[max(step-1, 0)]
    return steps.iloc[-1]

def pick_last_step(steps):
    return steps.iloc[-1]

def pick_step(steps, n=8, mode='min_l2'):
    if mode == 'last':
        return pick_last_step(steps)
    if mode == 'nth':
        return pick_nth_step(steps, n=n)
    if mode =='early':
        return early_stop_pred(steps, n=n)
    if mode == 'min_l2':
        return steps.loc[steps['postaction_l2_distance'].idxmin()]
    if mode == 'max_iou':
        return steps.loc[steps['postaction_iou'].idxmax()]
    if mode == 'max_coverage':
        return steps.loc[steps['postaction_coverage'].idxmax()]


def evaluate(input_path, tasks=None, mode='nth', n=10):


    with h5py.File(input_path, 'r') as dataset:
        keys = list(dataset.keys())
        print(f'Evaluating {input_path} with {len(keys)} keys')

        metrics = [
                    'deformable_distance',
                    'rigid_distance',
                    'weighted_distance',
                    'l2_distance',
                    'iou',
                    'coverage'
        ]

        stat_keys = [
            'episode_length',
            'out_of_frame',
            'nonadaptive_scale',
            'rotation',
            'scale',
            'percent_fling',
            'predicted_value',
            'predicted_deformable_value',
            'predicted_rigid_value',
            'deformable_weight',
        ]
        
        difficulties = ['easy', 'hard', 'none', 'flat', 'pick']
        rewards = ['deformable', 'rigid', 'weighted']

        step_df = []
        for k in tqdm(keys, desc='Reading keys...'):

            group_data = {}
            group = dataset.get(k)

            if tasks is not None:
                if np.sum(group['init_verts']) not in tasks:
                    continue

            episode = int(k.split('_')[0])
            step = int(k.split('_')[1][4:])

            level = str(group.attrs['task_difficulty'])     
            
            if level != 'hard':
                continue

            group_data['episode'] = episode
            group_data['step'] = step
            group_data['level'] = level
            group_data['key'] = str(k)
            group_data['input_path'] = input_path
            group_data['init_verts'] = np.sum(group['init_verts'])

            for key, value in group.attrs.items():
                group_data[key] = value

            step_df.append(group_data)

        
        step_df = pd.DataFrame(step_df)

        LEGACY_METRICS_MAP = {
            'deformable_distance': 'l2_distance',
            'rigid_distance': 'icp_distance',
            'weighted_distance': 'weighted_distance',
            'l2_distance': 'pointwise_distance',
            'coverage': 'coverage',
            'iou': 'iou',
        }


        retvals = []

        NUM_STEPS = 10
        for i in tqdm(range(NUM_STEPS), desc='Calculating stats for each step...'):
            retval = {}

            episode_df = pd.DataFrame()

            unique_episodes = step_df['episode'].unique()
            for episode in unique_episodes:
                #get nth step or the latest step
                picked_step = pick_step(step_df[step_df.episode == episode], mode='max_iou', n=i)
                episode_df = episode_df.append(picked_step, ignore_index=True)


            for difficulty in difficulties:
                if difficulty not in episode_df['task_difficulty'].unique():
                    continue

                for metric in metrics:
                    retval[f'combined/{difficulty}/{metric}/mean'] = (episode_df[episode_df.task_difficulty == difficulty][f'postaction_{LEGACY_METRICS_MAP[metric]}'].mean())
                    retval[f'combined/{difficulty}/{metric}/stderr'] = (episode_df[episode_df.task_difficulty == difficulty][f'postaction_{LEGACY_METRICS_MAP[metric]}'].std() / np.sqrt(len(episode_df[episode_df.task_difficulty == difficulty])))

                    if i == NUM_STEPS - 1:
                        print(f'{difficulty}/{metric}:{retval[f"combined/{difficulty}/{metric}/mean"]:.3f} +- {retval[f"combined/{difficulty}/{metric}/stderr"]:.3f}')
            
            if i == NUM_STEPS - 1:
                retval['episode_df'] = episode_df

            retval['deformable_weight'] = step_df.deformable_weight.mean()
            
            retvals.append(retval)

        return retvals, episode_df
    
if "__main__" == __name__:

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)

    args = parser.parse_args()

    retvals, episode_df = evaluate(args.input_path)

