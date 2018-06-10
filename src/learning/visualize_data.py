import matplotlib.pyplot as plt
from itertools import izip

# if running in the
# The default python provided in (Ana)Conda is not a framework build. However,
# the Conda developers have made it easy to install a framework build in both
# the main environment and in Conda envs. To use this install python.app conda
# install python.app and use pythonw rather than python


def load_data_from_file():
    import h5py
    with h5py.File('./data/costdata2d_10k.hdf5', 'r') as f:
        datasets = f['mydataset'][:]
    return datasets


def draw_one_data_point(fig, occ, sdf, cost, numb_rows=1, row=0):

    ax0 = fig.add_subplot(numb_rows, 3, 1 + 3 * row)
    image_0 = plt.imshow(occ)
    ax1 = fig.add_subplot(numb_rows, 3, 2 + 3 * row)
    image_1 = plt.imshow(sdf)
    ax2 = fig.add_subplot(numb_rows, 3, 3 + 3 * row)
    image_2 = plt.imshow(cost)

    draw_fontsize = 5

    ax0.tick_params(labelsize=draw_fontsize)
    ax1.tick_params(labelsize=draw_fontsize)
    ax2.tick_params(labelsize=draw_fontsize)

    if row == 0:
        ax0.set_title('Occupancy', fontsize=draw_fontsize)
        ax1.set_title('Signed Distance Field', fontsize=draw_fontsize)
        ax2.set_title('Chomp Cost', fontsize=draw_fontsize)

# This function draws two images next to each other.


def draw_grids(data):
    fig = plt.figure(figsize=(5, 2))
    draw_one_data_point(fig, data[0], data[1], data[2])
    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)

    # raw_input("Press [enter] to continue.")
    plt.close(fig)

if __name__ == '__main__':

    datasets = load_data_from_file()
    print "Data is now loaded !!!"
    print " -- datasets.shape : ", datasets.shape
    print " -- displaying with matplotlib..."
    for data1, data2, data3, data4 in izip(*[iter(datasets)] * 4):
        fig = plt.figure(figsize=(5, 6))
        draw_one_data_point(fig, data1[0], data1[1], data1[2], 4, 0)
        draw_one_data_point(fig, data2[0], data2[1], data2[2], 4, 1)
        draw_one_data_point(fig, data3[0], data3[1], data3[2], 4, 2)
        draw_one_data_point(fig, data4[0], data4[1], data4[2], 4, 3)
        plt.show(block=False)
        plt.draw()
        plt.pause(0.0001)

        # raw_input("Press [enter] to continue.")
        plt.close(fig)
