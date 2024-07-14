import pickle

# Define data
RET = 1.3102230553328593
MTX = [[2.88868277e+03, 0.00000000e+00, 1.46732299e+03],
       [0.00000000e+00, 2.89333800e+03, 2.02451966e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
DIST = [[1.58630260e-01, -1.08880580e+00, -1.76805105e-03, -8.12556620e-03],
        [3.15235533e+00]]

# Create a dictionary to store the data
data = {"RET": RET, "MTX": MTX, "DIST": DIST}

# Save the data to a pickle file
with open("calibration/calibration.pkl", "wb") as f:
    pickle.dump(data, f)

print("Calibration data saved to calibration.pckl")
