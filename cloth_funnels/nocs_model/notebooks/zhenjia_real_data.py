#%%
import matplotlib.pyplot as plt
import pickle

#%%
model = 'ours' # flingbot, ours, pp
cloth = 'blue' # blue, dress, red, pink
id = 0 # [0, 9]
data = pickle.load(open(f'/proj/crv/zhenjia/cloth-real-data/final-{model}/test-real:{cloth}{id}/test_data.pkl', 'rb'))
step = 4 # [0, 4]

#%%
init_color_img = data[step]['init_observation'][0]['color_img']
plt.imshow(init_color_img)

#%%
final_color_img = data[step]['final_observation'][0]['color_img']
plt.imshow(final_color_img)
# %%
