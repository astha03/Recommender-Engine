from recengine.src.data_processor import DataProcessor
from recengine.src.recommender_engine import RecommenderEngine


def main():
    data_proc = DataProcessor()
    data_proc.load_data("..\\data\\transactions.txt")
    data_proc.drop_small_orders(min_count=20)
    data_proc.one_hot_encode_columns(["l1"])
    data_proc.drop_columns(["l1", "l2", "l3", "sku", "brand", "UNCATEGORIZED"])
    data_proc.aggregate_orders()
    print("Preparing training data")
    data_proc.prepare_train_test_sets()
    training_data = data_proc.get_training_set()
    users_to_evaluate = data_proc.get_users_to_evaluate()

    rec_engine = RecommenderEngine(data=training_data, num_recs=15)
    print("Creating similarity matrix")
    rec_engine.create_similarity_matrix()
    print("Creating scores for users")
    rec_engine.score_users()
    print("Creating recommendations for users")
    rec_engine.generate_recommendations()
    print("Generating recommendations.csv")
    rec_engine.save_recommendations("..\\generated\\recommendations.csv")
    print(rec_engine.evaluate_rec_engine(users_to_evaluate))

    # Results:
    # Hit rate based on recommeder engine: 0.8034
    # Hit rate based on most popular items: 0.7982


if __name__ == "__main__":
    main()
