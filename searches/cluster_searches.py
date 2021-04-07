from pyspark.sql.functions import col, first, lit
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from searches.user_searches import user_genre_scores


def get_cluster_model_silhouette(cluster_model):
    """ Returns the silhouette score for a given model for a given dataframe.
    Model must be generated using user_cluster_model function """

    return cluster_model.sihlouette_score


def user_cluster_model(spark, ratings, movies, k, genres):
    """ Returns a clustering model for users' genre preferences """

    # Get all user ids
    all_user_ids = ratings.select(
        "userId").distinct().rdd.flatMap(lambda x: x).collect()

    # Calculate scores for each user
    scores = user_genre_scores(spark, ratings, movies, all_user_ids)\
        .sort(col("userId"), col("genre"))

    # Convert genres in rows to columns
    scores = scores.groupBy("userId").pivot(
        "genre").agg(first("score")).na.fill(0)

    # Ignore movies without genres
    if "(no genres listed)" in scores.columns:
        scores = scores.drop("(no genres listed)")
    scores.cache()

    # Find genres in dataset used
    genres_in_scores = scores.drop("userId").columns

    # Train a k-means model
    scores = VectorAssembler(
        inputCols=genres_in_scores, outputCol="features").transform(scores)
    kmeans_model = KMeans().setK(k).setSeed(5052).fit(scores)

    # Save genres used in model to model object
    kmeans_model.genres = genres_in_scores

    # Calculate sihlouette score & save to model
    train_predictions = kmeans_model.transform(scores)
    kmeans_model.sihlouette_score = ClusteringEvaluator().evaluate(train_predictions)

    return kmeans_model


def user_cluster_model_auto_k(spark, ratings, movies, max_k, genres):
    """ Returns a (model, best_k, dictionary) tuple; model is the k-means clustering
    model with the highest silhouette score for all values of k up to max_k; best_k is
    the value of k used for model; dictionary is a map between the values of k tested and
    the silhouette scores of the corresponding model"""

    k_to_score_dict = {}
    highest_score_k = None
    highest_score_model = None

    # Generate model for each k
    for k in range(2, max_k + 1):
        # Train model
        this_model = user_cluster_model(spark, ratings, movies, k, genres)

        # Add score to dictionary
        k_to_score_dict[k] = this_model.sihlouette_score

        if highest_score_model is None or \
                this_model.sihlouette_score > highest_score_model.sihlouette_score:
            # New high score => Save model
            highest_score_k = k
            highest_score_model = this_model

    return (highest_score_model, highest_score_k, k_to_score_dict)


def get_cluster_model_centroids(cluster_model):
    """ Returns a formatted string detailing the centroids in cluster_model """

    # Get centers & genres in model
    centers = cluster_model.clusterCenters()
    genres = cluster_model.genres

    # Build string
    out_string = ""
    for i in range(len(centers)):
        out_string += "\n  Cluster centroid " + str(i) + ":"
        for j in range(len(genres)):
            out_string += "\n    - " + genres[j] + ": " + str(centers[i][j])
        out_string += "\n"

    return out_string


def get_cluster_model_centroids_csv(cluster_model, csv_output):
    """ Returns csv formatted centroid locations """

    genres = cluster_model.genres
    centers = cluster_model.clusterCenters()

    output = open(csv_output + ".csv", "w")
    output.write(",".join(genres)+"\n")
    for i in range(len(centers)):

        for j in range(len(genres)):

            output.write(str(centers[i][j]))

            if j < len(genres) - 1:
                output.write(",")

        output.write("\n")
    output.close()


def get_users_cluster_predictions(spark, ratings, movies, cluster_model, user_ids):
    """ Returns the predicted cluster that each ID in user_ids is in for cluster_model  """

    # Get scores for given users
    users_scores = user_genre_scores(spark, ratings, movies, user_ids)

    # Convert genres in rows to columns
    users_scores = users_scores.groupBy("userId").pivot(
        "genre").agg(first("score")).na.fill(0).cache()

    # Add missing genres used in model & remove "no genres" genre
    genres_in_scores = users_scores.drop("userId").columns
    if "(no genres listed)" in users_scores.columns:
        genres_in_scores.drop("(no genres listed)")

    missing_cols = list(set(cluster_model.genres) -
                        set(genres_in_scores))

    for column in missing_cols:
        users_scores = users_scores.withColumn(column, lit(0))

    # Sort genres alphabetically
    users_scores = users_scores.select(
        sorted(users_scores.columns)).sort(col("userId"))

    # Use model to cluster given users
    users_scores = VectorAssembler(
        inputCols=cluster_model.genres, outputCol="features").transform(users_scores)

    predictions = cluster_model.transform(users_scores)\
        .select(["userId", "prediction"])\
        .withColumnRenamed("prediction", "predictedCluster")

    return predictions
