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
    ax = fig.add_subplot(gs[0, 1:4])
    ax.axis('off')
    img = np.array(group['action_visualization']).astype(np.uint8)
    ax.imshow(img[:, :, :3])

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    img = np.array(group['next_observations'])
    img = (img.transpose(1, 2, 0)*255).astype(np.uint8)
    ax.imshow(img[:, :, :3])
    output_path = path_prefix + ' f_before_after.png'
    plt.tight_layout(pad=0)
    plt.savefig(dir_path+output_path)
    plt.close(fig)
    episode = int(key.split("_")[0])
    step = int(key.split("_")[1][-2:])
    return f'<td><div>Episode {episode} Step {step}</div> \
        <img src="{output_path}" height="256px"> </td> '


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
    prefix = os.path.basename(os.path.dirname(path)) + '_'
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

            num_track = 10

            typical_episodes = []

            def stoep(key):
                return key.split('_')[0]

        
            for i, k in enumerate(tqdm(keys)):

                ep = stoep(k)
                if int(ep) > 10 and int(ep) < 15 and (ep not in [tup[1] for tup in typical_episodes]):
                    typical_episodes.append((None, ep))


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

            output += "<tr> <td> </td> <td> <h4> Regular Episode Performances </h4> </td> </tr>"
            typical_steps = []
            for r, ep in tqdm(typical_episodes):
                typical_steps = get_steps(ep)
                output = log_steps(output, typical_steps, '#f6d4ff')
