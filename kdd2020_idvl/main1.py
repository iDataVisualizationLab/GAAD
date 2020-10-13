from utils import *
from gcn import *
from sklearn.model_selection import train_test_split


def get_train_val_test(labels, seed=None, train_size=0.5, val_size=0.5):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    idx_train, idx_val = train_test_split(idx, random_state=None, train_size=train_size, test_size=val_size,
                                          stratify=labels)

    return idx_train, idx_val


if __name__ == "__main__":
    adj, features, labels = load_ndata(['../data/adj.pkl', '../data/feature.npy', '../data/train.npy'])
    # labels = labels - 1  # minus one for label index, in case we use 19 classes
    labels[labels==1] = labels[labels==1] - 1
    labels[labels>1] = labels[labels>1] - 2
    idx_train, idx_val = get_train_val_test(labels, train_size=0.7, val_size=0.3)

    dims = [100, 256, 128, 64, 18]

    # dims = [100, 256, 128, 64, 32, 18]

    model = GCN_norm(len(dims) - 1, dims)

    model.fit(features, adj, labels, idx_train, idx_val, train_iters=4000, verbose=True, patience=50)
