import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import (
    info_message,
    warning_message,
    debug_message
)

# from mlbmi import SentinelAOI


def kmeans_spatial_cluster(
        image, n_clusters=5, quantile_range=(1, 99),
        verbose=False, verbose_plot=False,
        scene_id=None, res=None, date=None):
    """Compute kmeans clustering spatially over image as grey scale levels

    Args:
        image (np.array): input greayscale image
        n_clusters (int, optional): number of grey scales. Defaults to 5.
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).

    Returns:
        sklearn.cluster._kmeans.KMeans: trained kmeans clustering object
    """
    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    pixel_scaled = sclr.fit_transform(image.reshape(-1, 1))

    # Configure kmeans clustering instance
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm='auto',
    )

    # Compute the K-Means clusters and store in object
    kmeans.fit(pixel_scaled)

    if verbose_plot:
        # Show BMI cluster images
        sanity_check_spatial_kmeans(
            kmeans, image, quantile_range=quantile_range,
            scene_id=scene_id, res=res, date=date
        )

    return kmeans


def kmeans_temporal_cluster(
        image_stack, n_clusters=5, quantile_range=(1, 99),
        verbose=False, verbose_plot=False,
        scene_id=None, res=None):
    """Compute kmeans clustering spatially over image as grey scale levels

    Args:
        image (np.array): input greayscale image
        n_clusters (int, optional): number of grey scales. Defaults to 5.
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).

    Returns:
        sklearn.cluster._kmeans.KMeans: trained kmeans clustering object
    """

    # Preprocess image data into a sequence of (nonzero) pixels over time
    samples_ = image_stack.reshape(image_stack.shape[0], -1).T
    where_zero = samples_.sum(axis=1) == 0
    samples_ = samples_[~where_zero]

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    samples_scaled = sclr.fit_transform(samples_)

    # Configure kmeans clustering instance
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm='auto',
    )

    # Compute the K-Means clusters and store in object
    kmeans.fit(samples_scaled)

    if verbose_plot:
        # Show BMI cluster images
        sanity_check_temporal_kmeans(
            kmeans, image_stack, quantile_range=quantile_range,
            scene_id=scene_id, res=res
        )

    return kmeans




def sanity_check_spatial_kmeans(kmeans, image, quantile_range=(1, 99),
                                scene_id=None, res=None, date=None,
                                plot_now=False):
    """Plot imshow of clustering solution as sanity check

    Args:
        kmeans (sklearn.cluster._kmeans.kmeans): object storing kmeans solution
        image (np.array): image with which kmeans was trains
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).
    """
    # Preprocess image data
    sclr = RobustScaler(quantile_range=quantile_range)
    pixel_scaled = sclr.fit_transform(image.reshape(-1, 1))

    # Predict each cluster value per pixel
    cluster_pred = kmeans.predict(pixel_scaled)

    base_fig_size = 5  # Each sub figure will be base_fig_size x base_fig_size
    fig, axs = plt.subplots(
        ncols=kmeans.n_clusters + 1,
        figsize=(base_fig_size * (kmeans.n_clusters + 1), base_fig_size)
    )

    # Plot the entire cluster_pred image
    axs[0].imshow(cluster_pred.reshape(image.shape), interpolation='None')

    # Cycle through and plot each cluster_pred image per 'class'
    for k in range(kmeans.n_clusters):
        axs[k + 1].imshow(
            (cluster_pred == k).reshape(image.shape),
            interpolation='None'
        )

    # Remove all unnecessary markers from figure
    [ax.grid(False) for ax in axs.ravel()]  # remove grid for images
    [ax.xaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks
    [ax.yaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks

    # Adjust figure to maximize use of gui box
    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.9,
        wspace=1e-2
    )

    # Set title for entire figure
    fig.suptitle(
        f"Spatial K-Means Reconstruction: {scene_id} - {res} - {date}",
        fontsize=20
    )

    if plot_now:
        # User can override default behaviour and plot on-the-fly
        plt.show()


def sanity_check_temporal_kmeans(
        kmeans, image_stack, quantile_range=(1, 99),
        scene_id=None, res=None, plot_now=False):
    """Plot imshow of clustering solution as sanity check

    Args:
        kmeans (sklearn.cluster._kmeans.kmeans): object storing kmeans solution
        image_stack (np.array): image_stack with which kmeans was trains
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).
    """
    samples_ = image_stack.reshape(image_stack.shape[0], -1).T
    where_zero = samples_.sum(axis=1) == 0
    samples_notzero = samples_[~where_zero]

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    samples_scaled = sclr.fit_transform(samples_notzero)

    # Predict each cluster value per pixel
    cluster_pred = kmeans.predict(samples_scaled)

    # Embedd above image in a zero array to re-constitute zeros
    #   for the out of mask shape
    cluster_image = np.zeros(samples_.shape[0])

    # Add one to each Class to represent the "out of mask" is class zero
    cluster_image[~where_zero] = cluster_pred + 1

    # Reshape 1D array into 2D image of the original image shape
    img_shape = image_stack.shape[1:]
    cluster_image = cluster_image.reshape(img_shape)

    base_fig_size = 5  # Each sub figure will be base_fig_size x base_fig_size
    fig, axs = plt.subplots(
        ncols=kmeans.n_clusters + 2,
        figsize=(base_fig_size * (kmeans.n_clusters + 1), base_fig_size)
    )

    # Plot the entire cluster_pred image
    axs[0].imshow(cluster_image, interpolation='None')

    # Plot the pixels outside the mask, which were not clustered
    axs[1].imshow(cluster_image == 0, interpolation='None')

    # Cycle through and plot each cluster_pred image per 'class'
    for k in range(kmeans.n_clusters):
        axs[k + 2].imshow((cluster_image == (k + 1)), interpolation='None')

    # Remove all unnecessary markers from figure
    [ax.grid(False) for ax in axs.ravel()]  # remove grid for images
    [ax.xaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks
    [ax.yaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks

    # Adjust figure to maximize use of gui box
    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.9,
        wspace=1e-2
    )

    # Set title for entire figure
    fig.suptitle(
        f"Temporal K-Means Reconstruction: {scene_id} - {res}",
        fontsize=20
    )

    if plot_now:
        # User can override default behaviour and plot on-the-fly
        plt.show()
