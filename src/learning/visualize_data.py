import matplotlib.pyplot as plt

def LoadData():
    import h5py
    with h5py.File('./data/costdata2d_10k.hdf5', 'r') as f:
        datasets = f['mydataset'][:]
    return datasets

# This function draws two images next to each other.    
def DrawGrids(occ, sdf, cost):
    fig = plt.figure(figsize=(5, 2))

    ax0 = fig.add_subplot(1, 3, 1)
    image_0 = plt.imshow(occ)
    ax1 = fig.add_subplot(1, 3, 2)
    image_1 = plt.imshow(sdf)
    ax2 = fig.add_subplot(1, 3, 3)
    image_2 = plt.imshow(cost)

    draw_fontsize=5

    ax0.tick_params(labelsize=draw_fontsize)
    ax1.tick_params(labelsize=draw_fontsize)
    ax2.tick_params(labelsize=draw_fontsize)

    ax0.set_title('Occupancy', fontsize=draw_fontsize)
    ax1.set_title('Signed Distance Field', fontsize=draw_fontsize)
    ax2.set_title('Chomp Cost', fontsize=draw_fontsize)

    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)
    
    # raw_input("Press [enter] to continue.")
    plt.close(fig)

if __name__ == '__main__':

    datasets = LoadData()
    for data in datasets:
        DrawGrids(data[0], data[1], data[2])
