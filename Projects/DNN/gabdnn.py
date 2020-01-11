# -*- coding: utf-8 -*-

# from __context__ import *
from __future__ import print_function, unicode_literals, division

import pandas as pd
import numpy as np
# import pickle
# from datetime import datetime
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorlayer as tl
# from IPython.core.display import clear_output

# In[3]:


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def compile_2d_data(rawdf, window=3, lookforward=1, enlarge=False, clean=True, verbose=False):

    if clean:
        df = rawdf.loc[rawdf['status'] == 0, :].dropna(how='all')
    else:
        df = rawdf.dropna(how='all')

    if df.shape[0] <= window + lookforward:
        return np.array([]), np.array([])

    df_pivot = df.pivot(index='ts', columns='code', values=['open', 'high', 'low', 'new_price', 'new_volume'])
    delta = df_pivot.pct_change().stack()[['open', 'high', 'low', 'new_price', 'new_volume']].values
    rtn = df_pivot.loc[:, 'new_price'].pct_change(lookforward).stack().dropna().values

    if enlarge:
        X = (10000 * (delta + 1)).astype(int)
        y = (10000 * (rtn + 1)).astype(int)
    else:
        X, y = delta, rtn

    # Generat snapshots with targets
    X_snaps = rolling_window(X.T, window).transpose(1, 2, 0)
    X_snaps = X_snaps.reshape(X_snaps.shape + (1,))[:-1]
    y_target = y[window:]
    assert X_snaps.shape[0] == y_target.shape[0]
    if verbose:
        print('Before rolling: ', X.shape, y.shape)
        print('After rolling: ', X_snaps.shape, y_target.shape)

    return X_snaps, y_target


def compile_3d_data(rawdf, window=3, lookforward=1, enlarge=False, clean=True, verbose=False):

    if clean:
        df = rawdf.loc[rawdf['status'] == 0, :].dropna(how='all')
    else:
        df = rawdf.dropna(how='all')

    df_pivot = df.pivot(index='ts', columns='code', values=['open', 'high', 'low', 'new_price', 'new_volume'])
    delta = df_pivot.pct_change().stack()[['open', 'high', 'low', 'new_price', 'new_volume']].values
    rtn = df_pivot.loc[:, 'new_price'].pct_change(lookforward).stack().dropna().values

    # Forming the third dimension
    k = delta[:, 4]
    k = k.repeat(4).reshape(k.shape + (4, 1,))
    arr = delta[:, :4]
    arr = arr.reshape(arr.shape + (1, ))
    new_x = np.dstack([arr, k]).transpose(0, 2, 1)

    if enlarge:
        X = (10000 * (new_x + 1)).astype(int)
        y = (10000 * (rtn + 1)).astype(int)
    else:
        X, y = new_x, rtn

    # Generat market snapshots with targets
    X_snaps = rolling_window(X.T, window).transpose(2, 3, 1, 0)[:-1]
    y_target = y[window:]
    assert X_snaps.shape[0] == y_target.shape[0]
    if verbose:
        print('Before rolling: ', X.shape, y.shape)
        print('After rolling: ', X_snaps.shape, y_target.shape)
    return X_snaps, y_target


def rebalance_by_class(data, method='max'):
    def __get_new_index(val_cnt):
        for k, v in val_cnt.items():
            tf_arr = (data == k)
            if np.sum(tf_arr) < v:
                out_idx0 = tf_arr.nonzero()[0]
                out_idx1 = np.random.choice(out_idx0, v - len(out_idx0), replace=True)
                out_idx = np.hstack([out_idx0, out_idx1])
            else:
                out_idx = np.random.choice(tf_arr.nonzero()[0], v, replace=False)
            np.random.shuffle(out_idx)
            yield [k, out_idx]

    # data = data_bin_y.copy()
    valcnt = pd.value_counts(data)
    # n_class = valcnt.shape[0]

    if method == 'max':
        out_cnt = {k: valcnt.max() for k, _ in valcnt.iteritems()}
    elif method == 'min':
        out_cnt = {k: valcnt.min() for k, _ in valcnt.iteritems()}
    elif method == 'average':
        out_cnt = {k: int(valcnt.mean()) for k, _ in valcnt.iteritems()}
    elif method == 'median':
        out_cnt = {k: int(valcnt.median()) for k, _ in valcnt.iteritems()}
    elif isinstance(method, dict):
        assert sum(method) == 1
        out_cnt = method
    else:
        raise ValueError('method is required.')

    return __get_new_index(out_cnt)


def batch_trimmer(X, y, batch_size=100, random_drop=True):
    assert X.shape[0] == y.shape[0]
    trim_n = (X.shape[0] // batch_size) * batch_size
    if random_drop:
        shuffle_idx = np.random.choice(range(len(X)), trim_n, replace=False)
        return X[shuffle_idx], y[shuffle_idx]
    else:
        return X[:trim_n], y[:trim_n]


def ml_split(data, ratios):
    assert sum(ratios) == 1
    cnt = np.shape(data)[0]
    slices = np.cumsum([int(cnt * r) for r in ratios])
    return np.split(data, slices)[:-1]


# In[] Model Evaluation
def plot_images(images, img_shape, cls_true, cls_pred=None):
    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def print_confusion_matrix(cls_pred, cls_true, num_classes):
    from sklearn.metrics import confusion_matrix

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_example_errors(images, img_shape, cls_true, cls_pred):
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.

    # Negate the boolean array.
    correct = cls_true == cls_pred
    incorrect = ~correct

    # Get the images from the test-set that have been
    # incorrectly classified.
    images_as_incorrect = images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true_of_incorrect = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=images_as_incorrect[0:9], img_shape=img_shape,
                cls_true=cls_true_of_incorrect[0:9],
                cls_pred=cls_pred[0:9])


def plot_weights(w, img_shape):
    # Get the values for the weights from the TensorFlow variable.

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    n = min(w.shape[1], 12)
    n_coln = min(n, 4)
    n_row = max(min(n // 4, 3), 1)
    fig, axes = plt.subplots(n_row, n_coln)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < n_coln * 3:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
