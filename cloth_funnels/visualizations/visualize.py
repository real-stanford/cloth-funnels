from h5py import File as HDF5
from matplotlib import pyplot as plt
import numpy as np
from environment.utils import visualize_grasp
import sys
from filelock import FileLock
import os
import seaborn as sns
import pandas as pd
from utils import collect_stats
from tqdm import tqdm
import pickle
from pprint import pprint
import json
import json
from jinja2 import Template
from learning.utils import rewards_from_group
import torch

def summarize(path):
    stats = collect_stats(path, int(1e7))
    for key, value in stats.items():
        if all(word not in key for word in ['distribution', 'img',
                                            'min', 'max', '_steps']):
            print(f'\t[{key:<36}]:\t{value:.04f}')
    # Episode lengths
    for key, value in stats.items():
        if "delta_flow" in key:
            difficulty = key.split('/')[1]
            break

    stats_dict = {}
    for key, value in stats.items():
        if 'distribution' in key:
            if 'episode' in key:
                print("distribution value shape", value.shape, np.sqrt(value.shape[0]))
            stats_dict[key] = {
                "mean": float(value.mean()),
                "std": float(value.std()),
                "stderr": float(value.std()/np.sqrt(value.shape[0])),
            }
    dir_name =  "/".join(path.split('/')[:-1])
    with open(dir_name + "/stats.json", "w") as outfile:
        json.dump(stats_dict, outfile)

   
    df = pd.DataFrame()
    averaged_coverages = []
    window = 10

    return None


def simple_visualize(group, key, path_prefix, dir_path):
    fig = plt.figure()
    fig.set_figheight(3.2)
    fig.set_figwidth(15)
    gs = fig.add_gridspec(1, 5)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    img = np.array(group['pretransform_observations'])
    img = (img.transpose(1, 2, 0)*255).astype(np.uint8)
    ax.imshow(img[:, :, :3].astype(np.uint8))
    ax.set_title(' Flow: {:.03f}, R: {:.03f}'.format(
        group.attrs['preaction_flow_correspondence'], group.attrs['preaction_flow_reward']))
    ax = fig.add_subplot(gs[0, 1:4])
    ax.axis('off')
    img = np.array(group['action_visualization']).astype(np.uint8)
    ax.imshow(img[:, :, :3])

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    img = np.array(group['next_observations'])
    img = (img.transpose(1, 2, 0)*255).astype(np.uint8)
    ax.imshow(img[:, :, :3])
    ax.set_title(' Flow: {:.03f}, R: {:.03f}'.format(
        group.attrs['postaction_flow_correspondence'], group.attrs['postaction_flow_reward']))

    output_path = path_prefix + ' f_before_after.png'
    plt.tight_layout(pad=0)
    plt.savefig(dir_path+output_path)
    plt.close(fig)
    episode = int(key.split("_")[0])
    step = int(key.split("_")[1][-2:])
    return f'<td><div>Episode {episode} Step {step}</div> \
        <div><strong> Preaction Distance </strong>{group.attrs["preaction_weighted_distance"]:.03f}</div> \
        <div> <strong>Postaction Distance </strong>{group.attrs["postaction_weighted_distance"]:.03f}</div> \
        <div><strong> Delta Weighted Dist </strong>{group.attrs["preaction_weighted_distance"] - group.attrs["postaction_weighted_distance"]:.3f} </div> </td><td>' +\
        f'<img src="{output_path}" height="256px"> </td> '


if __name__ == "__main__":
    path = sys.argv[1]
    with FileLock(path + '.lock'):
        with HDF5(path, 'r') as file:
            keys = []
            for k in file.keys():
                try:
                    keys.append(k)
                except:
                    pass
            print('keys:', len(keys))
    pprint(vars(pickle.load(
        open(path.split('replay_buffer.hdf5')[0] + 'args.pkl', 'rb'))))
    prefix = os.path.basename(os.path.dirname(path)) + '_'
    mean_final = summarize(path)
    # if input('visualize? (y/n)') != 'y':
    #     exit()
    dir_path = os.path.dirname(path) + '/'
    webpage_path = dir_path + 'index.html'
    #delete old html
    if os.path.exists(webpage_path):
        os.remove(webpage_path)

    for img_path in os.listdir(dir_path):
        if '.png' in img_path:
            os.remove(dir_path + img_path)
    
    print(f'Outputing visualizations to {webpage_path}')

    with FileLock(path + '.lock'):
        with HDF5(path, 'r') as file:
            use_simple_vis = 'all_obs' not in file[keys[0]]\
                or 'action_visualization' not in file[keys[0]]
            visualization_fn = simple_visualize\
                if use_simple_vis \
                else visualize_grasp
            output = """
                <head>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
                <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
            <script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.7/dist/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
                </head>
                <style>
                    table,
                    th,
                    td {
                        border: 1px solid black;
                        border-collapse: collapse;
                    }

                    .slidecontainer {
                        width: 100%;
                        /* Width of the outside container */
                    }

                    /* The slider itself */
                    .slider {
                        -webkit-appearance: none;
                        /* Override default CSS styles */
                        appearance: none;
                        width: 100%;
                        /* Full-width */
                        height: 25px;
                        /* Specified height */
                        background: #d3d3d3;
                        /* Grey background */
                        outline: none;
                        /* Remove outline */
                        opacity: 0.7;
                        /* Set transparency (for mouse-over effects on hover) */
                        -webkit-transition: .2s;
                        /* 0.2 seconds transition on hover */
                        transition: opacity .2s;
                    }

                    /* Mouse-over effects */
                    .slider:hover {
                        opacity: 1;
                        /* Fully shown on mouse-over */
                    }
                </style>
                <div class="slidecontainer">
                    <p>Speed</p>
                    <input type="range" min="1" max="10" value="5" class="slider" id="myRange">
                </div>
            """
            script = """
            <script>
                let slider = document.getElementById("myRange");

                function updateVideoSpeed(speed) {
                    let vids = document.getElementsByTagName('video')
                    // vids is an HTMLCollection
                    for (let i = 0; i < vids.length; i++) {
                        //#t=0.1
                        vids.item(i).playbackRate = speed;
                    }
                }
                updateVideoSpeed(slider.value)

                // Update the current slider value (each time you drag the slider handle)
                slider.oninput = function () {
                    updateVideoSpeed(this.value)
                }

            </script>
            """

            tmpl = Template('''
                <table>
                <h1>Summary</h1>
                {% for stat in stats %}
                <tr><td>{{ stat.split('/')[0] }}</td><td>{{ "%.3f &#177; %.3f"|format(stats[stat]["mean"], 2 * stats[stat]["stderr"]) }}</td><td>{{ "%.3f"|format(stats[stat]["std"]) }}</td></tr>
                {% endfor %}
                </table>
                ''')
            
            stats = json.load(open(dir_path + 'stats.json', 'r'))
            print(stats)
            output += tmpl.render(stats=stats)


            output += '<table style="width:100%">'

            num_track = 5
            max_episodes = [(-np.inf, 'blank_ep') for _ in range(num_track)]
            min_episodes = [(np.inf, 'blank_ep') for _ in range(num_track)]

            min_rewards = [(np.inf, 'blank_ep') for _ in range(num_track)]
            max_rewards = [(-np.inf, 'blank_ep') for _ in range(num_track)]

            max_prediction_errors = [(-np.inf, 'blank_ep') for _ in range(num_track)]
            min_prediction_errors = [(np.inf, 'blank_ep') for _ in range(num_track)]

            typical_episodes = []

            def stoep(key):
                return key.split('_')[0]

        
            for i, k in enumerate(tqdm(keys)):

                ep = stoep(k)
                delta_reward = rewards_from_group(file[k])['weighted']
                delta_reward = float(delta_reward)

                group = file[k]
                action_mask = torch.tensor(np.array(group['action_mask'])).bool()
                value_prediction = torch.tensor(np.array(group['value_map'])).masked_select(action_mask).item()

                prediction_error = abs(delta_reward - value_prediction)

                if int(ep) > 10 and int(ep) < 15 and (ep not in [tup[1] for tup in typical_episodes]):
                    typical_episodes.append((None, ep))

                if stoep(k) not in [tup[1] for tup in min_rewards]:
                    min_rewards.append((delta_reward, ep))
                    min_rewards.sort()
                    min_rewards = min_rewards[:num_track]

                if stoep(k) not in [tup[1] for tup in max_rewards]:
                    max_rewards.append((delta_reward, ep))
                    max_rewards.sort()
                    max_rewards = max_rewards[-num_track:]

                if stoep(k) not in [tup[1] for tup in max_prediction_errors]:
                    max_prediction_errors.append((prediction_error, ep))
                    max_prediction_errors.sort()
                    max_prediction_errors = max_prediction_errors[-num_track:]
                    
                if stoep(k) not in [tup[1] for tup in min_prediction_errors]:
                    min_prediction_errors.append((prediction_error, ep))
                    min_prediction_errors.sort()
                    min_prediction_errors = min_prediction_errors[:num_track]
                    
                if 'last' in k:
                    final_reward = -1 * (file[k].attrs['postaction_weighted_distance'] - file[k].attrs['init_weighted_distance'])
                    max_episodes.append((final_reward, ep))
                    max_episodes.sort()
                    max_episodes = max_episodes[-num_track:]
                    min_episodes.append((final_reward, ep))
                    min_episodes.sort()
                    min_episodes = min_episodes[:num_track]

            for r, ep in typical_episodes:
                print("typical episode:", ep, r)
            for r, ep in max_rewards:
                print("max reward", r, ep)
            for r, ep in min_rewards:
                print("min reward", r, ep)
            for r, ep in max_episodes:
                print("max episode", r, ep)
            for r, ep in min_episodes:
                print("min episode", r, ep)
            for r, ep in max_prediction_errors:
                print("max prediction error", r, ep)
            for r, ep in min_prediction_errors:
                print("min prediction error", r, ep)


            def get_steps(ep):
                k = ep
                out = []
                i = 0
                step = k + "_step{:02d}".format(i)
                while True:
                    if (step in keys):
                        out.append(step)
                    elif (step + "_last") in keys:
                        out.append(step + "_last")
                    else:
                        return out
                    i += 1
                    step = k + "_step{:02d}".format(i)

            def log_steps(output, steps, color=None):
                for k in steps:
                    style =  f"background-color:{color};"
                    if k in [k for _, k in max_rewards]:
                        style += "font-weight:bold;"
                    if k in [k for _, k in min_rewards]:
                        style += "font-style:italic;"
                    # if stoep(k) in [tup[1] for tup in max_prediction_errors]:
                    #     style += "font-weight:bold;"
                    
                    output += f'<tr style="{style}">'
                    group = file.get(k)

                    output += visualization_fn(
                        group=group,
                        key=k,
                        path_prefix=prefix + k,
                        dir_path=dir_path)
                    output += '</tr>'
                    with open(webpage_path, 'w') as webpage:
                        webpage.write(output + '</table>' + script)
                return output



            max_steps, min_steps, max_reward_steps, min_reward_steps = [], [], [], []
           
            # with open(webpage_path, 'w') as webpage:
            #         webpage.write(output + '</table>' + script)


            output += "<tr> <td> </td> <td> <h4> Regular Episode Performances </h4> </td> </tr>"
            typical_steps = []
            for r, ep in tqdm(typical_episodes):
                typical_steps = get_steps(ep)
                output = log_steps(output, typical_steps, '#f6d4ff')

            output += "<tr> <td> </td> <td> <h4> Maximum episode performances (max final score)</h4> </td> </tr>"
            for r, ep in tqdm(max_episodes):
                max_steps = get_steps(ep)
                output = log_steps(output, max_steps, '#ffd4d4')

            output += "<tr> <td> </td> <td>  <h4> Minimum episode performances (min final score)</h4> </td> </tr>"
            min_steps = []
            for r, ep in tqdm(min_episodes):
                min_steps = get_steps(ep)
                output = log_steps(output, min_steps, '#d6d4ff')

            output += "<tr> <td> </td> <td> <h4> Maximum prediction distances </h4> </td> </tr>"
            for r, ep in tqdm(max_prediction_errors):
                max_reward_steps = get_steps(ep)
                output = log_steps(output, max_reward_steps, '#ffd4d4')
            
            output += "<tr> <td> </td> <td> <h4> Minimum prediction distances </h4> </td> </tr>"
            for r, ep in tqdm(min_prediction_errors):
                min_reward_steps = get_steps(ep)
                output = log_steps(output, min_reward_steps, '#d6d4ff')
            
            # output += "<tr> <td> </td> <td> <h4> Maximum rewards </h4> </td> </tr>"
            # max_reward_steps = []
            # for r, ep in tqdm(max_rewards):
            #     max_reward_steps = get_steps(ep)
            #     output = log_steps(output, max_reward_steps, '#d4ffd5')

            # output += "<tr> <td> </td> <td> <h4> Minimum rewards </h4> </td> </tr>"
            # min_reward_steps = []
            # for r, ep in tqdm(min_rewards):
            #     min_reward_steps = get_steps(ep)
            #     output = log_steps(output, min_reward_steps, '#f6d4ff')

          
