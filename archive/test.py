import h5py

# Open the HDF5 file in read mode
filename = 'data.h5'
with h5py.File(filename, 'r') as f:
    # Print the keys at the root level of the HDF5 file
    print("Keys at the root level:")
    print(list(f.keys()))

    # Access specific datasets or groups within the file
    if 'tot_rm_dl' in f:
        dataset = f['tot_rm_dl']  # Access the dataset
        print("\nContents of 'data' dataset:")
        print(dataset)
        values = dataset[()]  # Read the values in the dataset
        print(values)