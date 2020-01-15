import pandas as pd
import numpy as np
from tqdm import tqdm


class RecommenderEngine:
    """
    Implements Item-based Collaborative Filtering. Creates an item based similarity matrix based on items different
    users purchased and then recommends items based on what a user just purchased.
    Uses Jaccard similarity metric for building the similarity matrix as our data is binary, i.e. we only have
    information on whether an item was purchased or not. We don't have any ratings.
    """
    def __init__(self, data, num_recs):
        self._data = data
        self._similarity_matrix = []  # n X n dataframe for n item columns in _data
        self._user_scores = []
        self._recs = []
        self.num_recs = num_recs

    def _create_empty_similarity_matrix(self):
        columns = list(self._data.columns)
        return pd.DataFrame(index=columns, columns=columns)

    def _memoized_jaccard_similarity(self):
        """
        Implements dynamic programming (memoization) to avoid doing repeated work. Calculated values are stored in
        cache and calculations are done only if values are not in the cache already.
        """
        cache = {}

        def calculate_jaccard_similarity(row_idx, col_idx):
            """
            row_idx and col_idx are the row and column indices in the similarity matrix. If row_idx corresponds to item 1
            and col_idx corresponds to item 2, we need to calculate Jaccard similarity between item 1 and item 2.
            First get vectors for item 1 and item 2 from _data representing the purchases for item 1 and item 2 across
            all orders. For example, for 4 users in _data: [1011] and [1001]
            Next get indices of non-zero elements (i.e. orders where the item was bought): [0,2,3] and [0,3]
            Jaccard similarity between x and y = |intersection(x,y|/|union(x,y|
            So in this case intersection = {0,3} and union = {0,2,3}. So Jaccard similarity = 2/3 = 0.66
            """
            if row_idx in cache:
                non_zero_elements_1 = cache[row_idx]
            else:
                item_1 = self._data.iloc[:, row_idx]
                non_zero_elements_1 = set(np.nonzero(item_1)[0])
                cache[row_idx] = non_zero_elements_1
            if col_idx in cache:
                non_zero_elements_2 = cache[col_idx]
            else:
                item_2 = self._data.iloc[:, col_idx]
                non_zero_elements_2 = set(np.nonzero(item_2)[0])
                cache[col_idx] = non_zero_elements_2
            intersection_size = len(non_zero_elements_1.intersection(non_zero_elements_2))
            union_size = len(non_zero_elements_1.union(non_zero_elements_2))
            if union_size == 0:
                return 0
            else:
                return round(intersection_size / union_size, 4)

        return calculate_jaccard_similarity

    def create_similarity_matrix(self):
        """
        Calculate correlation between different items using Jaccard similarity metric and store results in similarity matrix
        For N items types, an N X N similarity matrix will be created.
        We only need to calculate elements on one side of the diagonal, as similarity between x and y is equal to the
        similarity between y and x.
        Also, similarity of item with itself is 1.
        """
        self._similarity_matrix = self._create_empty_similarity_matrix()
        item_count = self._similarity_matrix.shape[0]
        # Update progress bar every 5 seconds
        progress_bar = tqdm(total=item_count, mininterval=5)
        for row_idx in range(0, item_count):
            progress_bar.update()
            for col_idx in range(row_idx, item_count):
                if row_idx == col_idx:
                    self._similarity_matrix.iloc[row_idx, col_idx] = 1.0
                else:
                    memoized_jaccard = self._memoized_jaccard_similarity()
                    similarity = memoized_jaccard(row_idx, col_idx)
                    self._similarity_matrix.iloc[col_idx, row_idx] = self._similarity_matrix.iloc[row_idx, col_idx] = similarity

    def score_users(self, user_matrix=None):
        """
        Generate scores for all existing users by default, or for new users if user_matrix is passed in.
        The score for user 1 and item 1 is calculated by taking dot product of row for user 1 in _data (representing
        items purchased by user 1) with the row for item 1 in the similarity matrix (representing similarity of
        item 1 with other items).
        Resulting data frame has order number as index.
        """
        if user_matrix is None:
            self._user_scores = round(self._data.dot(self._similarity_matrix), 4)
            self._user_scores["order_number"] = self._data.index
        else:
            self._user_scores = round(user_matrix.dot(self._similarity_matrix), 4)
            self._user_scores["order_number"] = user_matrix.index
        self._user_scores.set_index("order_number", inplace=True)

    def generate_recommendations(self):
        """
        Generate top N recommendations for each user and store result in _recs. Recommendations are stored in columns
        Rec_0 to Rec_N. Both the name of the item and the score are stored each Rec_i column.
        """
        users = self._data.index
        columns = ['Rec_' + str(x) for x in range(0, self.num_recs)]
        self._recs = pd.DataFrame(index=users, columns=columns)
        progress_bar = tqdm(total=len(users), mininterval=5)
        for user in users:
            progress_bar.update()
            scores_for_user = self._user_scores.loc[user, :]
            sorted_scores_for_user = scores_for_user.sort_values(ascending=False)
            for i in range(0, self.num_recs):
                score = sorted_scores_for_user.iloc[i]
                rec = sorted_scores_for_user.index[i]
                col_name = "Rec_" + str(i)
                self._recs.loc[user, col_name] = [rec, score]

    def evaluate_rec_engine(self, users_to_evaluate):
        """
        Compute hit rate based on recommendations by rec engine and based on most popular items.
        See DataProcessor.prepare_train_test_sets() for more details.
        """
        hit_rate_re = 0
        hit_rate_popular = 0
        popular_items = list(self._data.sum(axis=0).sort_values(ascending=False).head(self.num_recs).index)
        for order_num, removed_items in users_to_evaluate.items():
            # Remove the score for the item, as we just want the item name
            recommended_items = list(self._recs.loc[order_num].apply(lambda x: x[0]))
            for item in removed_items:
                if item in recommended_items:
                    hit_rate_re += 1
                if item in popular_items:
                    hit_rate_popular += 1

        hit_rate_re = round(hit_rate_re/len(users_to_evaluate), 4)
        hit_rate_popular = round(hit_rate_popular/len(users_to_evaluate), 4)

        result = "Hit rate based on recommeder engine: " + str(
            hit_rate_re) + "\nHit rate based on most popular items: " + str(hit_rate_popular)
        return result

    def save_recommendations(self, filename):
        self._recs.to_csv(filename)

    def get_similarity_matrix(self):
        return self._similarity_matrix

    def get_user_scores(self):
        return self._user_scores

    def get_data(self):
        return self._data

    def get_recommendations(self):
        return self._recs
