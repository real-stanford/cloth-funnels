import os
import sys
sys.path.append('/home/alper/folding-unfolding/src/PyFlex/bindings/build')
os.environ['PYFLEXROOT'] = '/home/alper/folding-unfolding/src/PyFlex'
os.environ['PYTHONPATH'] = '/home/alper/folding-unfolding/src/PyFlex/bindings/build'
os.environ['LD_LIBRARY_PATH'] = '/home/alper/folding-unfolding/src/PyFlex/external/SDL2-2.0.4/lib/x64'

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
from learning.nocs_unet_inference import get_flow_correspondence
import trimesh
import copy
import open3d as o3d
from pc_vis import *
from learning.utils import deformable_distance


def simple_visualize(group, key, path_prefix, dir_path, preaction_pc_path, postaction_pc_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pre_output_path = path_prefix + ' f_before.png'
    post_output_path = path_prefix + ' f_after.png'


    ax.axis('off')
    img = np.array(group['pretransform_observations'])
    img = (img.transpose(1, 2, 0)*255).astype(np.uint8)
    ax.imshow(img[:, :, :3].astype(np.uint8))
    plt.savefig(dir_path + pre_output_path)
    plt.close()



    fig, ax = plt.subplots(1, 1, figsize=(30, 10))

    action_vis_output_path = path_prefix + ' action_vis.png'
    ax.axis('off')
    img = np.array(group['action_visualization']).astype(np.uint8)
    ax.imshow(img[:, :, :3])
    plt.savefig(dir_path + action_vis_output_path)

    plt.close()


    # img = np.array(group['action_visualization']).astype(np.uint8)
    # ax.imshow(img[:, :, :3])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')
    img = np.array(group['next_observations'])
    img = (img.transpose(1,2,0)*255).astype(np.uint8)
    ax.imshow(img[:, :, :3].astype(np.uint8))
    plt.savefig(dir_path + post_output_path)
    plt.close()

   
    # plt.tight_layout(pad=0)
    # plt.savefig(dir_path+output_path)
    # plt.close(fig)
    print("action vis path: ", action_vis_output_path)
    return f'<td>' +\
        f'<img src="{pre_output_path}" height="256px"><img src="{preaction_pc_path}" height="256px"> </td> ' + f'<td>' +\
        f'<img src="{action_vis_output_path}" height="256px"> </td> ' + f'<td>' +\
        f'<img src="{post_output_path}" height="256px"><img src="{postaction_pc_path}" height="256px"> </td> '


if __name__ == "__main__":
    path = sys.argv[1]
    with FileLock(path + '.lock'):
        with HDF5(path, 'r') as file:
            keys = []
            for k in file.keys():
                try:
                    # file[k].attrs['max_coverage']
                    keys.append(k)
                except:
                    pass
            print('keys:', len(keys))
    # pprint(vars(pickle.load(
    #     open(path.split('replay_buffer.hdf5')[0] + 'args.pkl', 'rb'))))
    prefix = os.path.basename(os.path.dirname(path)) + '_'
    dir_path = os.path.dirname(path) + '/'
    webpage_path = dir_path + 'index.html'
    #delete old html
    if os.path.exists(webpage_path):
        os.remove(webpage_path)
    
    print(f'Outputing visualizations to {webpage_path}')

    # print("Keys", keys[0])
    with FileLock(path + '.lock'):
        with HDF5(path, 'r') as file:
            use_simple_vis = 'all_obs' not in file[keys[0]]\
                or 'action_visualization' not in file[keys[0]]
            visualization_fn = simple_visualize
            output = """
    <head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    </head>
    <style>

        body{
            
        }

        # table,
        th,
        td {
            border: 0.5px solid grey;
            border-collapse: collapse;
            border-radius: 5px;
            padding: 0px;
        }

        img {
            border-radius: 5px;
        }

        .green{
            color:green;
        }

        .red{
            color:red;
        }

        .stat-table {
            width: 100% !important;
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
    <body>
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
              <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.7/dist/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
            
            """

            tmpl = Template('''
                <table>
                <h1>Summary</h1>
                {% for stat in stats %}
                <tr><td>{{ stat.split('/')[0] }}</td><td>{{ "%.3f &#177; %.3f"|format(stats[stat]["mean"], 2 * stats[stat]["stderr"]) }}</td><td>{{ "%.3f"|format(stats[stat]["std"]) }}</td></tr>
                {% endfor %}
                </table>
                ''')
            
            # stats = json.load(open(dir_path + 'stats.json', 'r'))
            # output += tmpl.render(stats=stats)


            output += '<table style="width:100%" class="table">'

            episodes = []

            for k in tqdm(keys):
                episodes.append((file[k].attrs['rewards'], k))

            def get_steps(k):
                k = k.split('_')[0]
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

            def Ry(theta):
                return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                                [ 0           , 1, 0           ],
                                [-np.sin(theta), 0, np.cos(theta)]])

            def log_steps(output, steps, color=None):
                renderer = Renderer(512,512)
                for k in steps:
                    for t in range(1):
                        style =  f"background-color:#FFF;"
                        output += f'<tr style="{style}">'
                        group = file.get(k)

                    
                        init_verts = np.array(group['init_verts'])
                        preaction_verts =  np.array(group['preaction_verts'])
                        postaction_verts =  np.array(group['postaction_verts'])

                        # if t < 2:
                        #     random_trans = (np.random.random(size=(3, )) * 0.5) - 0.25
                        #     random_trans[1] = 0
                        #     preaction_verts =  np.array(group['preaction_verts']) + random_trans
                        #     postaction_verts =  np.array(group['postaction_verts']) + random_trans
                        # elif t < 5:
                        #     random_rotation = Ry(np.random.random() * 360)
                        #     preaction_verts =  np.array(group['preaction_verts']) @ random_rotation
                        #     postaction_verts = np.array(group['postaction_verts']) @ random_rotation


                        metrics = {}
                        for key, v in group.attrs.items():
                            metrics[key] = v

                        metrics['init_translation'] = np.mean(group['init_verts'], axis=0)


                        preaction_flow = get_flow_correspondence(np.array(group['pre_garmentnets_supervision_target']), metrics['preaction_coverage'])
                        metrics['preaction_flow_correspondence'] = preaction_flow['flow_correspondence']
                        metrics['preaction_direction'] = preaction_flow['direction']
                        metrics['preaction_translation'] = np.linalg.norm(np.mean(group['preaction_verts'], axis=0)-metrics['init_translation'])

                        beta = 0.2
                        metrics['preaction_enhanced_flow_correspondence'] = metrics['preaction_flow_correspondence'] * (1 - beta) + beta * ( 1 - np.clip(metrics['preaction_translation'], 0, 1))


                        postaction_flow = get_flow_correspondence(np.array(group['garmentnets_supervision_target']), metrics['postaction_coverage'])
                        metrics['postaction_flow_correspondence'] = postaction_flow['flow_correspondence']
                        metrics['postaction_direction'] = postaction_flow['direction']
                        metrics['postaction_translation'] = np.linalg.norm(np.mean(group['postaction_verts'], axis=0)-metrics['init_translation'])

                        metrics['postaction_enhanced_flow_correspondence'] = metrics['postaction_flow_correspondence'] * (1 - beta) + beta * ( 1 - np.clip(metrics['postaction_translation'], 0, 1))


                        preaction_weighted_distance, preaction_l2_distance, \
                            preaction_icp_cost, _, preaction_clouds = deformable_distance(init_verts, preaction_verts, group.attrs['max_coverage'])
                        postaction_weighted_distance, postaction_l2_distance, \
                            postaction_icp_cost, _, postaction_clouds = deformable_distance(init_verts, postaction_verts, group.attrs['max_coverage'])

                        metrics['preaction_weighted_distance'] = preaction_weighted_distance
                        metrics['preaction_l2_dist'] = preaction_l2_distance
                        metrics['preaction_icp_cost'] = preaction_icp_cost

                        metrics['postaction_weighted_distance'] = postaction_weighted_distance
                        metrics['postaction_l2_dist'] = postaction_l2_distance
                        metrics['postaction_icp_cost'] = postaction_icp_cost

                        # metrics['postaction_l2_dist'] =  np.min([p1, p2])
                        init_vert_cloud = preaction_clouds['init_vert_cloud']
                        preaction_verts_cloud = preaction_clouds['verts_cloud']
                        postaction_verts_cloud = postaction_clouds['verts_cloud']
                        icp_preaction_verts_cloud = preaction_clouds['icp_verts_cloud']
                        icp_postaction_verts_cloud = postaction_clouds['icp_verts_cloud']

                        preaction_reverse_init_verts_cloud = preaction_clouds['reverse_init_verts_cloud']
                        postaction_reverse_init_verts_cloud = postaction_clouds['reverse_init_verts_cloud']

                        preaction_pc_fig = f"{k}_preaction_l2_dist_{t}.png"
                        preaction_pc_path = os.path.join(dir_path, preaction_pc_fig)
                        postaction_pc_fig = f"{k}_postaction_l2_dist_{t}.png"
                        postaction_pc_path = os.path.join(dir_path, postaction_pc_fig)

                        plt.subplots(1, 3, figsize=(15, 5))

                        img = renderer.render_point_cloud(preaction_verts_cloud + preaction_reverse_init_verts_cloud, 
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05])
                        img = np.rot90(img, 1)
                        plt.subplot(1, 3, 1)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.imshow(img)


                        img = renderer.render_point_cloud(init_vert_cloud + preaction_verts_cloud, 
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05])
                        img = np.rot90(img, 1)
                        plt.subplot(1, 3, 2)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.imshow(img)

                        img = renderer.render_point_cloud(init_vert_cloud + icp_preaction_verts_cloud,
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05])
                        img = np.rot90(img, 1)
                        plt.subplot(1, 3, 3)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.imshow(img)
                        plt.savefig(preaction_pc_path)
                        plt.close()
                        
                        plt.subplots(1, 3, figsize=(10, 5))

                        img = renderer.render_point_cloud(postaction_verts_cloud + postaction_reverse_init_verts_cloud,
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05])
                        img = np.rot90(img, 1)  
                        plt.subplot(1, 3, 1)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.imshow(img)

                        img = renderer.render_point_cloud(init_vert_cloud + postaction_verts_cloud,
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05])
                        img = np.rot90(img, 1)  
                        plt.subplot(1, 3, 2)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.imshow(img)

                        img = renderer.render_point_cloud(init_vert_cloud + icp_postaction_verts_cloud,
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05])
                        img = np.rot90(img, 1)
                        plt.subplot(1, 3, 3)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.imshow(img)
                        plt.savefig(postaction_pc_path)

                        plt.close()

                        output += visualization_fn(
                            group=group,
                            key=k,
                            path_prefix=prefix + k,
                            dir_path=dir_path,
                            preaction_pc_path = preaction_pc_fig,
                            postaction_pc_path = postaction_pc_fig)
                        output += '</tr>'
                    
                        output += f'<tr style="{style}">'

                        metrics['preaction_l1_dist'] =  np.mean(np.linalg.norm(init_verts - preaction_verts, axis=1, ord=1))
                        metrics['postaction_l1_dist'] =  np.mean(np.linalg.norm(init_verts - postaction_verts, axis=1, ord=1))

                        metrics['preaction_weighted_dist'] = preaction_weighted_distance
                        metrics['postaction_weighted_dist'] = postaction_weighted_distance
                    
                        tmpl = Template('''
                        <td style="padding:none">
                        <table class="stat-table">
                        <tr> <td> Coverage: </td> <td>{{ "%.3f"|format(attrs[prefix + '_coverage']) }} </td> </tr>
                        <tr> <td> Flow: </td> <td>{{ "%.3f"|format(attrs[prefix + '_flow_correspondence']) }} </td> </tr>
                        <tr> <td style="color:#0F0;"> L2 Distance: </td> <td>{{ "%.3f"|format(attrs[prefix + '_l2_dist']) }} </td> </tr>
                        <tr> <td> L1 Distance: </td> <td>{{ "%.3f"|format(attrs[prefix + '_l1_dist']) }} </td> </tr>
                        <tr> <td> Direction: </td> <td>{{ "%.3f"|format(attrs[prefix + '_direction']) }} </td> </tr>
                        <tr> <td> Translation: </td> <td>{{ "%.3f"|format(attrs[prefix + '_translation']) }} </td> </tr>
                        <tr> <td style="color:#00F;"> ICP Cost: </td> <td>{{ "%.3f"|format(attrs[prefix + '_icp_cost']) }} </td> </tr>
                        </table>
                        </td>
                        ''')

                        mid_tmpl = Template('''
                        <td style="padding:none">
                        <table class="stat-table">
                        <tr> <td> Delta Coverage: </td> <td>{{ "%.3f"|format(attrs['postaction_coverage'] - attrs['preaction_coverage']) }} </td> </tr>
                        <tr> <td {% if (attrs['postaction_flow_correspondence'] - attrs['preaction_flow_correspondence']) > 0 %} class="green" {% else %} class="red" {% endif %}> Delta Flow: </td> <td>{{ "%.3f"|format(attrs['postaction_flow_correspondence'] - attrs['preaction_flow_correspondence']) }} </td> </tr>
                        <tr> <td {% if (attrs['postaction_enhanced_flow_correspondence'] - attrs['preaction_enhanced_flow_correspondence']) > 0 %} class="green" {% else %} class="red" {% endif %}> Delta T + Flow: </td> <td>{{ "%.3f"|format(attrs['postaction_enhanced_flow_correspondence'] - attrs['preaction_enhanced_flow_correspondence']) }} </td> </tr>
                        <tr> <td> Final Flow: </td> <td>{{ "%.3f"|format(attrs['postaction_flow_correspondence']) }} </td> </tr>
                        <tr> <td {% if (attrs['postaction_l2_dist'] - attrs['preaction_l2_dist']) < 0 %} class="green" {% else %} class="red" {% endif %}> Delta L2 Reward: </td> <td>{{ "%.3f"|format(-1 * (attrs['postaction_l2_dist'] - attrs['preaction_l2_dist'])) }} </td> </tr>
                        <tr> <td {% if (attrs['postaction_icp_cost'] - attrs['preaction_icp_cost']) < 0 %} class="green" {% else %} class="red" {% endif %}> Delta ICP Reward: </td> <td>{{ "%.3f"|format(-1 * (attrs['postaction_icp_cost'] - attrs['preaction_icp_cost'])) }} </td> </tr>
                        <tr> <td {% if (attrs['postaction_weighted_dist'] - attrs['preaction_weighted_dist']) < 0 %} class="green" {% else %} class="red" {% endif %}> Delta Weighted Reward: </td> <td>{{ "%.3f"|format(-1 * (attrs['postaction_weighted_dist'] - attrs['preaction_weighted_dist'])) }} </td> </tr>
                        <tr> <td> Final L2 Reward: </td> <td>{{ "%.3f"|format(-1 * attrs['postaction_l2_dist'])}} </td> </tr>
                        <tr> <td> Final ICP Reward: </td> <td>{{ "%.3f"|format(-1 * attrs['postaction_icp_cost'])}} </td> </tr>
                        <tr> <td> Final Weighted Reward: </td> <td>{{ "%.3f"|format(-1 * attrs['postaction_weighted_dist'])}} </td> </tr>

                        </table>
                        </td>
                        ''')

                        output += tmpl.render(prefix="preaction", attrs=metrics)
                        output += mid_tmpl.render(attrs=metrics)
                        output += tmpl.render(prefix="postaction", attrs=metrics)
                        output += '</tr>'
                        output += f'</tr>'

                        with open(webpage_path, 'w') as webpage:
                            webpage.write(output + '</table>' + script)


                return output

            for i, (r, ep) in tqdm(enumerate(episodes)):
                output = log_steps(output, [ep], '#ffd4d4')

            output += '</body>'
            with open(webpage_path, 'w') as webpage:
                webpage.write(output)
          
