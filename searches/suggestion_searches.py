from searches.user_searches import user_genre_scores
from pyspark.sql import types
from pyspark.sql.functions import col, expr, first, udf, array, lit
from math import sqrt
from pyspark.ml.feature import VectorAssembler


def scores_df_euclidean_dist(dataframe):
    """ Calculates the euclidean distance between the two
    scores encoded in the dataframe. Should not be used outside of the
    nearest_neighbours function, as it is heavily coupled. """

    distance = 0
    for i in range(len(dataframe[0])):
        distance += (dataframe[0][i] - dataframe[1][i]) ** 2
    return sqrt(distance)


def nearest_neighbours(spark, ratings, movies, user_ids):
    """ Retrieves the distances to all other users from each user in user_ids, 
    returned in ascending order, with the euclidean distance between each user's 
    genre scores as the distance measure """

    # Register user-defined function with spark
    distance_func = udf(
        lambda x: scores_df_euclidean_dist(x), types.FloatType())
    spark.udf.register("distance_func", distance_func)

    # Get user ids
    all_user_ids = ratings.select(
        "userId").distinct().rdd.flatMap(lambda x: x).collect()

    # Calculate scores for each user
    scores = user_genre_scores(spark, ratings, movies, all_user_ids)\
        .sort(col("userId"), col("genre")).cache()

    # Convert genres in rows to columns
    scores = scores.groupBy("userId").pivot(
        "genre").agg(first("score")).na.fill(0)

    # Ignore movies without genres
    if "(no genres listed)" in scores.columns:
        scores = scores.drop("(no genres listed)")
    scores.cache()

    # Transform genre scores into arrays
    scores = VectorAssembler(
        inputCols=scores.drop("userId").columns,
        outputCol="genreScores").transform(scores)\
        .select(["userId", "genreScores"]).cache()

    # Dataframe to return
    user_neighbours = spark.createDataFrame(
        user_ids, types.StringType()).toDF("userId")\
        .withColumn("neighbourId", lit(None))\
        .withColumn("distance", lit(None))

    for user_id in user_ids:
        # Separate user's scores
        user_scores = scores.where(scores.userId == user_id)\
            .withColumnRenamed("genreScores", "userGenreScores")\
            .drop("userId")
        other_scores = scores.where(scores.userId != user_id)

        # Cross join user's scores for access in user-defined function
        other_scores = other_scores.crossJoin(user_scores)

        # Calculate distance to each datapoint from user's datapoint
        distances = other_scores.withColumn(
            "distance", distance_func(array("genreScores", "userGenreScores")))\
            .select(["userId", "distance"])\
            .withColumnRenamed("userId", "neighbourId")\
            .withColumn("userId", lit(user_id))\
            .sort(col("distance").asc())\
            .select(["userId", "neighbourId", "distance"])\
            .cache()

        # Join to dataframe to return
        user_neighbours = user_neighbours.union(distances)

    # Drop null values & return
    return user_neighbours.na.drop().cache()


def get_movie_suggestions(spark, ratings, movies, user_id, neighbour_user_ids):
    """ Provides suggested movies for a user (user_id), given their neighbours (neighbour_user_ids) """

    # Find highest positively (>2.5) rated movies for neighbour users
    neighbour_movies = ratings.where(ratings.userId.isin(neighbour_user_ids))\
        .join(movies, "movieId")\
        .withColumnRenamed("userId", "suggestedByUserId")\
        .sort(col("rating").desc())
    neighbour_movies = neighbour_movies.where(
        neighbour_movies.rating > 2.5)

    # Find movie ids for ratings made by given user
    users_ratings = [row.movieId for row in ratings.where(ratings.userId == user_id).collect()]

    # Remove movie suggestions that the user has already seen
    neighbour_movies = neighbour_movies.where(~neighbour_movies.movieId.isin(users_ratings))

    return neighbour_movies
