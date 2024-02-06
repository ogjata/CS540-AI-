from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    yaleData = np.load(filename)
    meanYaleData = np.mean(yaleData, axis=0)
    centered_yale_data= yaleData - meanYaleData

    return centered_yale_data

def get_covariance(dataset):
    # Your implementation goes here!
    n = dataset.shape[0]
    covariance_matrix = np.dot(np.transpose(dataset), dataset) / (n - 1)

    return covariance_matrix

def get_eig(S, m):
    # Your implementation goes here!
    eigenvalues, eigenvectors = eigh(S)

    sorted_indixes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indixes]
    eigenvectors = eigenvectors[:, sorted_indixes]

    eigenvalues_m = eigenvalues[:m]
    eigenvectors_m = eigenvectors[:, :m]

    return np.diag(eigenvalues_m), eigenvectors_m

def get_eig_prop(S, prop):
    # Your implementation goes here!
    eigenvalues, eigenvectors = eigh(S)

    sorted_indixes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indixes]
    eigenvectors = eigenvectors[:, sorted_indixes]

    variance = eigenvalues / np.sum(eigenvalues)

    m = np.sum(variance >= prop)
    
    eigenvalues_m = eigenvalues[:m]
    eigenvectors_m = eigenvectors[:, :m]

    return np.diag(eigenvalues_m), eigenvectors_m

def project_image(image, U):
    # Your implementation goes here!
    projection = np.dot(np.transpose(U), image)

    return np.dot(projection, np.transpose(U))

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    new_orig = orig.reshape(32, 32)
    new_proj = proj.reshape(32, 32)

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    ax1.set_title("Original")
    ax2.set_title("Projection")

    ax1.imshow(new_orig, aspect='equal')
    ax2.imshow(new_proj, aspect='equal')
    cbar1 = fig.colorbar(ax1.imshow(new_orig, aspect='equal'), ax=ax1)
    cbar2 = fig.colorbar(ax2.imshow(new_proj, aspect='equal'), ax=ax2)

    return fig, ax1, ax2