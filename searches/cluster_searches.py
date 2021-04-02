from pyspark.sql.functions import col, first, monotonically_increasing_id
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from searches.user_searches import user_genre_scores


def user_cluster_model(spark, ratings, movies, k, genres):

    # TODO - Return center point locations & number of users per cluster
    # TODO - Take in user_ids and return center points which they are in
    # TODO - Return "nearby" users in same cluster
    """ Returns a clustering model for users' genre preferences """  # TODO

    # Get scores for all users & sort alphabetically
    # TODO - Remove limit
    all_user_ids = ratings.select("userId").limit(
        1000).distinct().rdd.flatMap(lambda x: x).collect()

    # Calculate scores for each user
    scores = user_genre_scores(spark, ratings, movies, all_user_ids)\
        .sort(col("userId"), col("genre"))

    # Convert genres in rows to columns
    scores = scores.groupBy("userId").pivot(
        "genre").agg(first("score")).na.fill(0).cache()

    # Find genres in dataset used
    genres_in_scores = scores.drop("userId").columns
    if "(no genres listed)" in scores.columns:
        genres_in_scores.drop("(no genres listed)")

    # Train a k-means model
    scores = VectorAssembler(
        inputCols=genres_in_scores, outputCol="features").transform(scores)
    kmeans_model = KMeans().setK(3).setSeed(5052).fit(scores)  # TODO - Set k

    # Save genres used in model to model object
    kmeans_model.genres = genres_in_scores

    return kmeans_model
    # # Show centres
    # print("Cluster Centers: ")
    # for center in kmeans_model.clusterCenters():
    #     print("\tCENTER :", center, "\n")

    # centers = kmeans_model.clusterCenters()
    # for i in range(0, len(centers)):
    #     centers[i] = centers[i].tolist()

    # centers = spark.sparkContext.parallelize(centers)\
    #     .toDF(genres_in_scores)

    # centers = centers.withColumn("id", monotonically_increasing_id()).show()


def get_cluster_model_centers(cluster_model):
    """ Returns a formatted string detailing the centers in cluster_model """
    # TODO - test this function

    # Get centers & genres in model
    centers = cluster_model.clusterCenters()
    genres = cluster_model.genres

    # Build string
    out_string = ""
    for i in range(len(centers)):
        out_string += "\n\t Cluster" + str(i)
        for j in range(len(genres)):
            out_string += "\n\t " + genres[j] + ": " + str(centers[i][j])

    return out_string


def get_users_clusters(spark, ratings, movies, cluster_model, user_ids):
    """ Returns the predicted cluster that each ID in user_ids is in for cluster_model  """
    # TODO - Test this function

    # Get scores for given users
    users_scores = user_genre_scores(spark, ratings, movies, user_ids)

    # Convert genres in rows to columns
    users_scores = users_scores.groupBy("userId").pivot(
        "genre").agg(first("score")).na.fill(0).cache()

    # Add missing genres used in model
    genres_in_scores = users_scores.drop("userId").columns
    if "(no genres listed)" in users_scores.columns:
        genres_in_scores.drop("(no genres listed)")

    missing_cols = list(set(cluster_model.genres) -
                        set(genres_in_scores.columns))

    for column in missing_cols:
        users_scores = users_scores.withColumn(column, 0)

    # Sort genres alphabetically
    users_scores = users_scores.select(
        sorted(users_scores.columns)).sort(col("userId"))

    # Use model to predict cluster
    # TODO

    exit(0)

    # TODO - Get users scores
    # TODO - Find centroid for each users


def cluster_model_silhouette(cluster_model):
    """ Returns the silhouette score for a given model """

    exit()


def user_cluster_model_auto_k(spark, ratings, movies, max):
    """ Returns a dictionary of k-to-sihlouette scores by iterating through
    different values for k """

    exit(0)
