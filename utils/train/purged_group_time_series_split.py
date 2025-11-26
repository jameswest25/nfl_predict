# utils/train/purged_group_time_series_split.py
#
# As found here: https://github.com/scikit-learn-contrib/scikit-learn-extra/blob/main/sklearn_extra/model_selection/_split.py
# Included locally to avoid potential dependency issues and allow for customization.

import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator with non-overlapping, contiguous groups.

    - Produces tail-only, forward-in-time test folds: n_splits equal-sized,
      consecutive test blocks taken from the end of the series.
    - Training for fold i consists of all groups strictly before the test
      block, minus ``group_gap`` groups immediately preceding the test.
    - Optionally supports an explicit fixed ``test_group_size`` (in number of
      groups). If not provided, test size defaults to
      ``n_groups // (n_splits + 1)`` (capped by ``max_test_group_size``).
    - ``post_group_gap`` can be used to embargo groups immediately after the
      current fold's test block from the training set (more relevant for
      rolling-window variants; with tail-only folds it is typically a no-op).
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 test_group_size=None,
                 group_gap=0,
                     post_group_gap=0,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.test_group_size = test_group_size
        self.post_group_gap = post_group_gap
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1

        # Ensure groups are provided as a NumPy array
        groups = np.asarray(groups)

        # Validate that groups are contiguous blocks (no repeated non-contiguous chunks)
        # Find the start of each new group block
        block_starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        block_labels = groups[block_starts]
        # If any label appears more than once in block_labels, the groups are non-contiguous
        if block_labels.size != np.unique(block_labels).size:
            raise ValueError(
                "Groups must be contiguous blocks in the provided data order."
            )

        # Use contiguous block labels directly to preserve time order
        unique_groups = block_labels
        n_groups = unique_groups.size

        # Pre-compute group -> slice(start_idx, end_idx) to avoid repeated full-array scans
        block_ends = np.r_[block_starts[1:], n_samples]
        label_to_slice = {
            label: slice(int(start), int(end))
            for label, start, end in zip(unique_groups, block_starts, block_ends)
        }

        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}.").format(n_folds, n_groups))

        # Determine test group size with guard for tiny n_groups and optional explicit size
        if np.isinf(max_test_group_size):
            max_test_cap = n_groups
        else:
            max_test_cap = int(max_test_group_size)

        if self.test_group_size is not None:
            group_test_size = int(self.test_group_size)
            if group_test_size <= 0:
                raise ValueError("test_group_size must be a positive integer.")
            if group_test_size > max_test_cap:
                raise ValueError(
                    "test_group_size exceeds max_test_group_size cap."
                )
            if self.n_splits * group_test_size > n_groups:
                raise ValueError(
                    "Not enough groups for the requested n_splits and test_group_size."
                )
        else:
            group_test_size = min(n_groups // n_folds, max_test_cap)
            if group_test_size <= 0:
                raise ValueError(
                    "Not enough groups for the requested n_splits/max_test_group_size."
                )
        group_test_starts = list(range(n_groups - n_splits * group_test_size,
                                       n_groups, group_test_size))
        
        indices = np.arange(n_samples)

        for i, group_test_start in enumerate(group_test_starts):
            # Compute train window bounds, handling the no-cap case for max_train_group_size
            if np.isinf(max_train_group_size):
                train_start = 0
            else:
                train_start = max(0, int(group_test_start - group_gap - max_train_group_size))
            train_end = max(0, group_test_start - group_gap)
            if train_end <= train_start:
                # Empty train window; skip this fold
                continue
            train_groups = unique_groups[train_start:train_end]
            
            # Compute test groups for this fold
            test_groups = unique_groups[group_test_start : group_test_start + group_test_size]

            # Optional post-test embargo around the CURRENT test block
            test_end = group_test_start + group_test_size
            if self.post_group_gap:
                embargo_groups = unique_groups[
                    test_end : min(n_groups, test_end + int(self.post_group_gap))
                ]
                if embargo_groups.size:
                    train_groups = train_groups[~np.isin(train_groups, embargo_groups)]

            # Build indices by concatenating precomputed slices (avoids O(n) scans)
            if train_groups.size:
                train_index_parts = [
                    np.arange(label_to_slice[label].start, label_to_slice[label].stop)
                    for label in train_groups
                ]
                train_indices = (
                    np.concatenate(train_index_parts) if train_index_parts else np.array([], dtype=int)
                )
            else:
                train_indices = np.array([], dtype=int)

            if test_groups.size:
                test_index_parts = [
                    np.arange(label_to_slice[label].start, label_to_slice[label].stop)
                    for label in test_groups
                ]
                test_indices = (
                    np.concatenate(test_index_parts) if test_index_parts else np.array([], dtype=int)
                )
            else:
                test_indices = np.array([], dtype=int)

            if test_indices.size == 0 or train_indices.size == 0:
                # Skip empty windows
                continue

            if self.verbose > 0:
                    pass

            yield train_indices, test_indices 