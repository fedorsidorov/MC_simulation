import numpy as np

# %%
old_cnt = 0
now_cnt = 1

for i in range(112):
    old_data = np.load('/Volumes/Transcend/e_DATA_500nm_point/0/e_DATA_Pv_' + str(i) + '.npy')
    now_data = np.load('/Volumes/Transcend/e_DATA_500nm_point_NEW/e_DATA_Pn_' + str(i) + '.npy')
    # now_data = np.load('/Volumes/Transcend/e_DATA_500nm_point_NEW/0/e_DATA_Pv_' + str(i) + '.npy')

    old_cnt += len(np.where(
        np.logical_and(
            old_data[:, 3] == 1,
            old_data[:, 6] <= 500
        )
    )[0])
    now_cnt += len(np.where(
        np.logical_and(
            now_data[:, 3] == 1,
            now_data[:, 6] <= 500
        )
    )[0])


print(old_cnt / now_cnt)

# %%
np.where(now_data[:, 6] > 500)[0]
