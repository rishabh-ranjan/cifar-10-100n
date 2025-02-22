import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import torch.nn.functional as F


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root))
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root),
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


# basic function#
def multiclass_noisify(y, P, random_state=0):
    """Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print np.max(y), P.shape[0]
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print m
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    print(f"flip with random seed {random_state}")

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
    flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1.0 - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
    flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1.0 - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1.0 - n
        P[nb_classes - 1, nb_classes - 1] = 1.0 - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise


def noisify(
    dataset="mnist",
    nb_classes=10,
    train_labels=None,
    noise_type=None,
    noise_rate=0,
    random_state=0,
):
    if noise_type == "pairflip":
        train_noisy_labels, actual_noise_rate = noisify_pairflip(
            train_labels, noise_rate, random_state=0, nb_classes=nb_classes
        )
    if noise_type == "symmetric":
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(
            train_labels, noise_rate, random_state=0, nb_classes=nb_classes
        )
    return train_noisy_labels, actual_noise_rate


def noisify_instance(train_data, train_labels, noise_rate):
    if max(train_labels) > 10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q) == 50000:
            break

    w = np.random.normal(loc=0, scale=1, size=(32 * 32 * 3, num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample, w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(
            np.random.choice(np.arange(num_class), p=p_all / sum(p_all))
        )
    over_all_noise_rate = (
        1
        - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum()) / 50000
    )
    return noisy_labels, over_all_noise_rate


"""
def noisify_instance(train_data,train_labels,noise_rate):
    if max(train_labels)>10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    p_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    p = []
    for pro in p_:
        if 0 < pro < 1:
            p.append(pro)
        if len(p)==50000:
            break

    w = []
    for i in range(num_class):
        w.append(np.random.normal(loc=0,scale=1,size=3072))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        w_sample = w[train_labels[i]]
        dot = np.dot(sample,w_sample)
        norm_dot = 1/(1+np.exp(-dot))
        p_flip = p[i]*norm_dot
        p_random = np.random.rand()
        if p_random <= p_flip:
            random_label = np.random.choice(range(num_class),1)
            noisy_labels.append(int(random_label))
        else:
            noisy_labels.append(train_labels[i])
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    return noisy_labels, over_all_noise_rate
"""
