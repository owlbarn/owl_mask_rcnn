import h5py

# Converts Keras weight into a format easily readable by hdf5-caml.

fname = "mask_rcnn_coco.h5"
dfname = "mask_rcnn_coco_owl.hdf5"

f = h5py.File(fname, 'r')
data_file = h5py.File(dfname, 'w')

for node_name in f.keys():
    for subfolder in f[node_name].keys():
        for param in f[node_name][subfolder].keys():
            conv_weight = f[node_name][subfolder][param].value.tolist()
            data_file.create_dataset(subfolder + param, data=conv_weight)

f.close()
data_file.close()

