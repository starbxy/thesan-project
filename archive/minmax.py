import h5py

def find_min_max(filename):
    with h5py.File(filename, 'r') as file:
        # Assuming the dataset you want to analyze is named 'data'
        dataset = file['Bmag']

        # Calculate min and max values
        min_value = dataset[()].min()
        max_value = dataset[()].max()

        return min_value, max_value

# Usage example
filename = 'Bmag.hdf5'
min_value, max_value = find_min_max(filename)
print(f"Minimum value: {min_value}")
print(f"Maximum value: {max_value}")