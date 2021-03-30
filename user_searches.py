from pyspark.sql.functions import countDistinct, explode, expr, col, count
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import round as spark_round


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

    users = spark.createDataFrame(user_ids, "string").toDF("userId")

    # Join aggreations to users dataframe, removing duplicate userId columns
    users = users.join(user_ratings_agg, users.userId ==
                       user_ratings_agg.userId, "left").drop(user_ratings_agg.userId)

    # Replace null values
    users = users.na.fill(0, ["moviesWatched", "genresWatched"])

    return users


def user_genre_scores(spark, ratings, movies, user_ids):
    """ Returns a dataframe containing genre scores for users matching IDs in user_ids """

    # Find ratings made by given user
    filtered_ratings = ratings.where(ratings.userId.isin(user_ids))

    # Find associated movies to found ratings
    movies_ratings = filtered_ratings.join(movies, "movieId")

    # Expand genre arrays into multiple rows
    movies_ratings = movies_ratings.withColumn(
        "genres", explode(movies_ratings.genres))

    # Dataframe for output
    user_genre_scores = spark.createDataFrame(
        user_ids, "string").toDF("userId")

    # Find sum and count of ratings for each user
    scores = movies_ratings.groupBy('userId', 'genres').agg(
        count('rating'), spark_sum('rating')).\
        withColumnRenamed("count(rating)", "ratingCount").\
        withColumnRenamed("sum(rating)", "ratingSum")

    # Add one 5.0 and one 0.0 rating to aggregates
    scores = scores.withColumn("ratingCount", expr("ratingCount + 2")).\
        withColumn("ratingSum", expr("ratingSum + 5"))

    # Find mean rating for "score"
    scores = scores.withColumn("score", col("ratingSum") / col("ratingCount")).\
        drop(col("ratingCount")).drop("ratingSum")

    # Sort datframe by ID & score
    scores = scores.sort(col("userId").asc(), col("score").desc())

    # Round scores to 2 d.p.
    scores = scores.withColumn("score", spark_round(scores.score, 2))

    return scores
