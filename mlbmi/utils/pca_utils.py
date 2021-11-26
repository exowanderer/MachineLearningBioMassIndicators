import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, MiniBatchSparsePCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from . import (
    info_message,
    warning_message,
    debug_message
)

# from mlbmi.utils.base_utils import debug_message, warning_message, info_message
# from mlbmi import SentinelAOI


def determine_n_components_extra(image_size, n_components):
    for n_comp_extra in range(image_size):
        n_samples_ = image_size / (n_components + n_comp_extra)
        if int(n_samples_) == n_samples_:
            return n_comp_extra


def pca_spatial_components(
        image, n_components=5, quantile_range=(1, 99), whiten=True,
        verbose=False, verbose_plot=False, scene_id=None, res=None, date=None):
    """Compute pca components spatially over image as grey scale levels

    Args:
        image (np.array): input greayscale image
        n_components (int, optional): number of grey scales. Defaults to 5.
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).

    Returns:
        sklearn.components._pca.PCA: trained pca components object
    """

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    n_extra = determine_n_components_extra(image.size, n_components)
    sclr = RobustScaler(quantile_range=quantile_range)
    pixel_scaled = sclr.fit_transform(
        image.reshape(-1, n_components + n_extra))

    # Configure pca components instance
    pca = PCA(
        n_components=n_components + n_extra,
        copy=True,
        whiten=whiten,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=None,
    )

    # Compute the PCA components and store in object
    pca.fit(pixel_scaled.reshape(-1, n_components + n_extra))

    if verbose_plot:
        # Show BMI components images
        sanity_check_spatial_pca(
            pca, image, quantile_range=quantile_range,
            scene_id=scene_id, res=res, date=date
        )

    return pca


def pca_temporal_components(
        image_stack, n_components=5, quantile_range=(1, 99), whiten=True,
        verbose=False, verbose_plot=False, scene_id=None, res=None):
    """Compute pca components spatially over image as grey scale levels

    Args:
        image (np.array): input greayscale image
        n_components (int, optional): number of grey scales. Defaults to 5.
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).

    Returns:
        sklearn.components._pca.PCA: trained pca components object
    """

    # Preprocess image data into a sequence of (nonzero) pixels over time
    samples_ = image_stack.reshape(image_stack.shape[0], -1).T
    # where_zero = samples_.sum(axis=1) == 0
    # samples_ = samples_[~(samples_.sum(axis=1) == 0)]

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    samples_scaled = sclr.fit_transform(samples_)

    n_components_ = np.min([n_components, image_stack.shape[0]])
    if n_components_ != n_components:
        warning_message(
            "n_components cannot be larger than the number of time stamps.\n"
            f"Modified n_components from {n_components} to {n_components_}."
        )

    # Configure pca components instance
    pca = PCA(
        n_components=n_components_,
        copy=True,
        whiten=whiten,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=None,
    )

    # Compute the PCA components and store in object
    pca.fit(samples_scaled)

    if verbose_plot:
        # Show BMI components images
        sanity_check_temporal_pca(
            pca, image_stack, quantile_range=quantile_range,
            scene_id=scene_id, res=res
        )

    return pca


def sanity_check_spatial_pca(
        pca, image, quantile_range=(1, 99),
        scene_id=None, res=None, date=None,
        plot_now=False):
    """Plot imshow of components solution as sanity check

    Args:
        pca (sklearn.components._pca.pca): object storing pca solution
        image (np.array): image with which pca was trains
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).
    """
    # Preprocess image data
    # n_extra = determine_n_components_extra(image.size, pca.n_components)
    sclr = RobustScaler(quantile_range=quantile_range)
    pixel_scaled = sclr.fit_transform(
        image.reshape(-1, pca.n_components)
    )

    # Predict each components value per pixel
    components_pred = pca.transform(pixel_scaled)

    base_fig_size = 5  # Each sub figure will be base_fig_size x base_fig_size
    fig, axs = plt.subplots(
        ncols=pca.n_components + 1,
        figsize=(base_fig_size * (pca.n_components + 1), base_fig_size),
        sharex=True,
        sharey=True
    )

    # Plot the entire components_pred image
    axs[0].imshow(components_pred.reshape(image.shape), interpolation='None')

    # Cycle through and plot each components_pred image per 'class'
    for k, comp_ in enumerate(pca.components_):
        axs[k + 1].imshow(
            (components_pred == comp_).T.reshape(image.shape),
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
        f"Spatial PCA Reconstruction: {scene_id} - {res} - {date}",
        fontsize=20
    )

    if plot_now:
        # User can override default behaviour and plot on-the-fly
        plt.show()


def sanity_check_temporal_pca(
        pca, image_stack, quantile_range=(1, 99),
        scene_id=None, res=None, plot_now=False):
    """Plot imshow of components solution as sanity check

    Args:
        pca (sklearn.components._pca.pca): object storing pca solution
        image_stack (np.array): image_stack with which pca was trains
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).
    """
    samples_ = image_stack.reshape(image_stack.shape[0], -1).T
    # where_zero = samples_.sum(axis=1) == 0
    # samples_notzero = samples_[~where_zero]

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    # samples_scaled = sclr.fit_transform(samples_notzero)
    samples_scaled = sclr.fit_transform(samples_)

    # Predict each components value per pixel
    components_pred = pca.transform(samples_scaled)

    # Embedd above image in a zero array to re-constitute zeros
    #   for the out of mask shape
    # components_image = np.zeros(samples_.shape[0])
    image_shape = list(image_stack.shape[1:])
    image_shape.append(pca.n_components)
    components_image = components_pred.reshape(image_shape)

    # Add one to each Class to represent the "out of mask" is class zero
    # components_image[~where_zero] = components_pred + 1

    # Reshape 1D array into 2D image of the original image shape
    # img_shape = image_stack.shape[1:]
    # components_image = components_image.reshape(img_shape)

    base_fig_size = 5  # Each sub figure will be base_fig_size x base_fig_size
    fig, axs = plt.subplots(
        ncols=pca.n_components,
        figsize=(base_fig_size * (pca.n_components + 1), base_fig_size)
    )

    # # Plot the entire components_pred image
    # axs[0].imshow(components_image, interpolation='None')

    # Plot the pixels outside the mask, which were not componentsed
    # axs[1].imshow(components_image == 0, interpolation='None')

    # Cycle through and plot each components_pred image per 'class'
    for k in range(pca.n_components):
        axs[k].imshow(components_image[:, :, k], interpolation='None')

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
        f"Temporal PCA Reconstruction: {scene_id} - {res}",
        fontsize=20
    )

    if plot_now:
        # User can override default behaviour and plot on-the-fly
        plt.show()
