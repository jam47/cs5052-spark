from pyspark.sql import types
from pyspark.sql.functions import countDistinct, explode, expr, col, count, mean, stddev, row_number, first, collect_list, monotonically_increasing_id
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import round as spark_round
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler


def users_by_ids(spark, ratings, movies, user_ids):
    """ Returns a dataframe containing users matching provided IDs in user_ids """

    # Get ratings associated with users
    user_ratings = ratings.filter(ratings.userId.isin(user_ids))

    # Count distinct rated movie IDs for each user
    user_ratings_agg = user_ratings.groupBy("userId").agg(countDistinct(
        "movieId")).withColumnRenamed("count(movieId)", "moviesWatched")

    # Get movies associated with users via found ratings
    user_movies = user_ratings.join(movies, "movieId")

    # Explode genres for each movie
    user_movies = user_movies.withColumn(
        "genres", explode(user_movies.genres))

    # Count distinct genres for each user
    user_movies_agg = user_movies.groupBy("userId").agg(countDistinct(
        "genres")).withColumnRenamed("count(genres)", "genresWatched")

    # Summarise aggregations in new users datatframe via joins
    users = spark.createDataFrame(user_ids, types.StringType()).toDF("userId")
    users = users.join(user_ratings_agg, users.userId == user_ratings_agg.userId, "left")\
        .drop(user_ratings_agg.userId)\
        .join(user_movies_agg, users.userId == user_movies_agg.userId, "left")\
        .drop(user_movies_agg.userId)

    # Sort by ID
    users = users.withColumn("userId", users.userId.cast(types.IntegerType()))\
        .sort(col("userId").asc())

    # Replace null values
    users = users.na.fill(0, ["moviesWatched", "genresWatched"])

    return users


def user_genre_scores(spark, ratings, movies, user_ids):
    """ Returns a dataframe containing genre scores for users matching IDs in user_ids """

    # Find ratings made by given users
    filtered_ratings = ratings.where(ratings.userId.isin(user_ids))

    # Find associated movies to found ratings
    movies_ratings = filtered_ratings.join(movies, "movieId")

    # Expand genre arrays into multiple rows
    movies_ratings = movies_ratings.withColumn(
        "genres", explode(movies_ratings.genres))\
        .withColumnRenamed("genres", "genre")

    # Create dataframe for output
    user_genre_scores = spark.createDataFrame(
        user_ids, types.StringType()).toDF("userId")

    # Find sum and count of ratings for each user
    scores = movies_ratings.groupBy('userId', 'genre').agg(
        count('rating').alias("ratingCount"), spark_sum('rating').alias("ratingSum"))

    # Add one 5.0 and one 0.0 rating to aggregates
    scores = scores.withColumn("ratingCount", expr("ratingCount + 2")).\
        withColumn("ratingSum", expr("ratingSum + 5"))

    # Find mean rating for "score"
    scores = scores.withColumn("score", col("ratingSum") / col("ratingCount")).\
        drop(col("ratingCount")).drop("ratingSum")

    return scores


def formatted_user_genre_scores(spark, ratings, movies, user_ids):
    """ Returns a formatted dataframe containing genre scores for users matching IDs in user_ids """

    # Get scores
    scores = user_genre_scores(spark, ratings, movies, user_ids)

    # Sort dataframe by ID & score
    scores = scores.sort(col("userId").asc(), col("score").desc())

    # Round scores to 2 d.p.
    scores = scores.withColumn("score", spark_round(scores.score, 3))

    return scores


def user_taste_comparison(spark, ratings, movies, user_ids):
    """ Returns a string to be displayed to output which compares the
    taste of users matching IDs in user_ids"""

    # Find ratings made by given users
    users_ratings = ratings.where(ratings.userId.isin(user_ids))\
        .drop("timestamp")

    # Count each user's ratings
    users_ratings_counts = users_ratings.groupBy("userId")\
        .agg(count("rating").alias("numRatings"))

    # Standardise (scale) ratings for each user
    users_ratings_agg = users_ratings.groupBy("userId")\
        .agg(mean("rating").alias("avgRating"), stddev("rating").alias("stddevRating"))

    users_ratings = users_ratings.join(users_ratings_agg, "userId")\
        .withColumn("scaledRating", (col("rating") - col("avgRating") / col("stddevRating")))\
        .drop("avgRating").drop("stddevRating")

    # Format mean & standard dev to 2 d.p. for output
    users_ratings_agg = users_ratings_agg\
        .withColumn("avgRating", spark_round(users_ratings_agg.avgRating))\
        .withColumn("stddevRating", spark_round(users_ratings_agg.stddevRating))

    # Find highest & lowest rated movies on scaled average between users
    avg_movie_ratings = users_ratings.groupBy("movieId")\
        .agg(mean("scaledRating").alias("avgScaledRating"))\

    highest_rated_movie_id = avg_movie_ratings.sort(col("avgScaledRating").desc())\
        .first().movieId
    highest_rated_movie = movies.filter(
        movies.movieId == highest_rated_movie_id).first()

    lowest_rated_movie_id = avg_movie_ratings.sort(col("avgScaledRating").asc())\
        .first().movieId
    lowest_rated_movie = movies.filter(
        movies.movieId == lowest_rated_movie_id).first()

    # Find highest & lowest ranked genres on average
    genre_scores = user_genre_scores(spark, ratings, movies, user_ids)

    genre_ranks = genre_scores.withColumn("genreRank", row_number().over(
        Window.partitionBy("userId").orderBy("score")))

    avg_genre_ranks = genre_ranks.groupBy("genre").agg(
        mean("genreRank").alias("avgGenreRank"))

    highest_avg_genre = avg_genre_ranks.sort(col("avgGenreRank").desc())\
        .first().genre

    lowest_avg_genre = avg_genre_ranks.sort(col("avgGenreRank").asc())\
        .first().genre

    # Generate output string from gathered data
    output_string = "COMPARING USERS: " + " ".join(user_ids)

    output_string += "\n\n Number of ratings:"
    for user_id in user_ids:
        output_string += "\n\t User " + str(user_id) + ": "
        output_string += str(users_ratings_counts.filter(
            users_ratings_counts.userId == user_id).first().numRatings)

    output_string += "\n\n Mean rating:"
    for user_id in user_ids:
        output_string += "\n\t User " + str(user_id) + ": "
        output_string += str(users_ratings_agg.filter(
            users_ratings_agg.userId == user_id).first().avgRating)

    output_string += "\n\n Rating standard deviation:"
    for user_id in user_ids:
        output_string += "\n\t User " + str(user_id) + ": "
        output_string += str(users_ratings_agg.filter(
            users_ratings_agg.userId == user_id).first().stddevRating)

    output_string += "\n\n Highest rated movie between users: " + \
        highest_rated_movie.title + \
        " (ID: " + str(highest_rated_movie_id) + ")"

    output_string += "\n\n Lowest rated movie between users: " + \
        lowest_rated_movie.title + " (ID: " + str(lowest_rated_movie_id) + ")"

    output_string += "\n\n Highest ranked genre between users: " + \
        highest_avg_genre

    output_string += "\n\n Lowest ranked genre between users: " + \
        lowest_avg_genre

    return output_string
