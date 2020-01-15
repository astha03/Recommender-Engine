import pandas as pd
import numpy as np
import random
from tqdm import tqdm


class DataProcessor:
    def __init__(self):
        """
        _data is a dataframe with order number as index and items as columns
        _users_to_evaluate is a series with
            index=order number and
            values=list of items that were set to 0 in the training set
        """
        self._data = None
        self._training_set = None
        self._users_to_evaluate = pd.Series()

    def load_data(self, filename):
        try:
            self._data = pd.read_csv(filename, sep="\t")
        except FileNotFoundError as error:
            return "Unable to find file."

    def get_data(self):
        return self._data

    def get_training_set(self):
        return self._training_set

    def get_users_to_evaluate(self):
        return self._users_to_evaluate

    def drop_small_orders(self, order_column="order_number", min_count=2):
        """
        In the dataset, there is one row for each item purchased in an order. If an order has 10 items
        (of same or different type),there are 10 rows with the same order number. Quantities of items
        purchased in orders varies from 1 to 588. Drop orders that had less than min_count items purchased to
        make the data more manageable and also to get rid of less useful orders. Resulting dataframe has users
        that purchased at least min_count items of the same or different type.
        """
        grouped_by_order_number = self._data.groupby(order_column).size().reset_index(name="counts")
        filtered_orders = grouped_by_order_number[grouped_by_order_number["counts"] > min_count][order_column]
        self._data = self._data[self._data[order_column].isin(filtered_orders)]

    def one_hot_encode_columns(self, columns=[]):
        """
        One hot encoding is needed to convert the categorical values for the items to numerical values.
        In the resulting dataframe, each row represents one order, and each column represents an item.
        Value 1 means that item was purchased in that order, whereas value 0 means it was not purchased.
        """
        data_list = []
        data_list.append(self._data)
        for col in columns:
            transformed_col = pd.get_dummies(data=self._data[col])
            data_list.append(transformed_col)
        self._data = pd.concat(data_list, axis=1)  # axis=1 for columns

    def drop_columns(self, columns=[]):
        self._data.drop(columns, axis=1, inplace=True)

    def aggregate_orders(self, order_column="order_number"):
        """
        Aggregate the rows by grouping by the order number and taking the sum of the items purchased (i.e. sum the
        values in each column). Resulting dataframe has order number as the index (this ensures order number is unique)
        """
        self._data = self._data.groupby(order_column).sum()

    def prepare_train_test_sets(self):
        """
        Usually, for evaluation of an ML algorithm, the dataset is usually split into training and test sets.
        For evaluating a recommender engine, in production we would use all the data up to a certain point
        for creating the similarity matrix, make recommendations for those users and then compare those
        recommendations to actual purchases made by those users in the future. Since we do not have this type
        of data, we still need some way to evaluate our rec engine to get a sense of how well it is doing.
        Here, we will use hit rate.
        Evaluation strategy:
        1. Training set is first created as a copy of the original data set
        2. For each user(order number) that purchased more than one type of item in the training set, randomly remove one type of item
           (leave one out cross-validation), i.e. change value of one item column to 0 in the training set.
        3. For the modified users in step 2, store the items that were removed in _users_to_evaluate
        4. Use the training set to create a similarity matrix and generate recommendations for each user.
        5. For each modified user, if the removed item is in the list of recommended items, then it's a hit, otherwise it's a miss.
        6. To calculate hit rate, divide the sum of hits across all the modified users by the total number of modified users.
        7. Also create a list of the most popular items, and calculate hit rate based on whether or not the removed item is in this list.
        """
        training_set = self._data.copy()
        # Find indices non-zero items for each user. non-zero returns a tuple, so just get first element.
        non_zero_indices = training_set.apply(np.nonzero, axis=1).apply(lambda x: x[0])
        # Filter out users that purchased only one item
        non_zero_indices = non_zero_indices[non_zero_indices.apply(lambda x: len(x) > 1)]
        # List of tuples: first element in tuple is order number, second element is array of indices of non-zero elements
        user_non_zero_idx_pairs = list(zip(non_zero_indices.index, non_zero_indices))
        # Use same seed every time to get same results on each run
        random.seed(0)
        progress_bar = tqdm(total=len(user_non_zero_idx_pairs), mininterval=5)
        for pair in user_non_zero_idx_pairs:
            progress_bar.update()
            user_idx = pair[0]
            item_idx = list(pair[1])
            row = training_set.loc[user_idx]
            # num_sample = int(round(len(item_idx) * 0.2))
            num_sample = 1
            # Randomly sample one item and change it to 0 for each user. Store column names for modified items.
            sample_idx = random.sample(item_idx, num_sample)
            row[sample_idx] = 0
            column_name = training_set.columns[sample_idx]
            self._users_to_evaluate.loc[user_idx] = list(column_name)
        self._training_set = training_set