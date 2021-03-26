from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark.sql.functions import col, substring, expr, split, explode, mean, count, array_contains, countDistinct
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import round as spark_round
import sys
import argparse
from contextlib import redirect_stdout

# Env variables
APP_NAME = "CS5052 P1 - Apache Spark"
DATASET_FILEPATH = "data/ml-latest-small-modified/"

# Search for keys
USERS_SF = "users"
MOVIES_SF = "movies"
FAV_GENRES_SF = "fav_genres"

# Search by keys
USERS_SB = "user_ids"
MOVIE_IDS_SB = "movie_ids"
MOVIE_NAMES_SB = "movie_names"
YEARS_SB = "years"
GENRES_SB = "genres"
RATING_SB = "rating"
WATCHES_SB = "watches"


# ===================================
# ======== ARGUMENT PARSING =========
# ===================================
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Search for
    parser.add_argument("-f", "--search-for", action="store", dest="search_for", required=True,
                        choices=[USERS_SF, MOVIES_SF, FAV_GENRES_SF],
                        help=("The type of entity to search for:"
                              "\n\t- " + USERS_SF + ": Search for users"
                              "\n\t- " + MOVIES_SF + ": Search for movies"
                              "\n\t- "  # TODO
                              ))

    # Search by
    parser.add_argument("-b", "--search-by", action="store", dest="search_by", required=True,
                        choices=[USERS_SB, MOVIE_IDS_SB, MOVIE_NAMES_SB, GENRES_SB,
                                 YEARS_SB, RATING_SB, WATCHES_SB],
                        help=("The type of entity to search by:"
                              "\n\t- " + USERS_SB +
                              " : Search by user IDs (must supply value using -v)"
                              "\n\t- " + MOVIE_IDS_SB +
                              " : Search by movie IDs (must supply value using -v)"
                              "\n\t- " + MOVIE_NAMES_SB +
                              " : Search by movie names (must supply value using -v)"
                              "\n\t- " + GENRES_SB +
                              " : Search by genre names (must supply value using -v)"
                              "\n\t- " + RATING_SB + " : List top rated movies"
                              "\n\t- " + WATCHES_SB + " : List most watched movies"
                              ))

    # Search value
    parser.add_argument("-v", "--value", action="store", dest="search_value", nargs='+',
                        help="The value to search by")

    # Results length
    parser.add_argument("-c", "--count", action="store", dest="result_count", default=10, type=int,
                        help="The number of results to return")

    # Results as csv list
    parser.add_argument("-l", "--csv", dest="csv_out",
                        help="Output result as CSV to given filepath")

    # Output file
    parser.add_argument("-o", "--output", action="store", dest="outfile",
                        help="The output filepath for the program. If no file value is supplied, output will be written to stdout.")

    # Parse args & validate for-by search combination
    args = parser.parse_args()

    if args.search_for == USERS_SF and args.search_by != USERS_SB:
        parser.print_help()
        sys.exit(0)

        # TODO - Fav genre can only be searched by user ID

    # TODO - Enforce value for certain searches

    # TODO - Output provided values to output

    return args


# ===================================
# =========== MAIN METHOD ===========
# ===================================
def main(spark, args):
    # Open outfile
    if args.outfile is not None and args.outfile != "":
        output = open(args.outfile, "w")
    else:
        output = sys.stdout

    # Read dataset
    ratings = spark.read.csv(DATASET_FILEPATH + "/ratings.csv", header=True)
    movies = spark.read.csv(DATASET_FILEPATH + "/movies.csv", header=True)

    ratings = ratings.alias('ratings')
    movies = movies.alias('movies')

    # Persist dataset
    ratings.cache()
    movies.cache()
    # TODO - Cache others here

    # Split genres into array
    movies = movies.withColumn("genres", split(
        "genres", "\|"))

    # Separate years from movie titles
    movies = movies.withColumn("year", substring(col("title"), -5, 4))
    movies = movies.withColumn("title", expr(
        "substring(title, 1, length(title)-7)"))

    # Cast numerical columns
    ratings = ratings.withColumn(
        "rating", ratings.rating.cast(types.FloatType()))

    # Perform search
    if args.search_for == USERS_SF:
        if args.search_by == USERS_SB:
            # Search for users by IDs
            users_by_ids(ratings, movies, args.search_value,
                         args.csv_out, output)

    elif args.search_for == FAV_GENRES_SF:
        if args.search_by == USERS_SB:
            # Search for users' favourite genres
            find_favourite_genre(
                ratings, movies, args.search_value, args.result_count, args.csv_out, output)

    elif args.search_for == MOVIES_SF:
        if args.search_by == MOVIE_IDS_SB:
            # Search for movies by IDs
            movies_by_ids(ratings, movies, args.search_value,
                          args.csv_out, output)

        elif args.search_by == MOVIE_NAMES_SB:
            # Search for movies by names
            movies_by_names(ratings, movies, args.search_value, args.csv_out,
                            output)

        elif args.search_by == GENRES_SB:
            # Search for movies by genres
            movies_by_genres(ratings, movies, args.search_value,
                             args.result_count, args.csv_out, output)

        elif args.search_by == USERS_SB:
            # Search for movies by user IDs
            movies_by_user_ids(
                ratings, movies, args.search_value, args.result_count, args.csv_out, output)

        elif args.search_by == YEARS_SB:
            # Search for movies by year
            movies_by_years(ratings, movies, args.search_value,
                            args.result_count, args.csv_out, output)

        elif args.search_by == RATING_SB:
            # List movies by highest rating
            movies_by_rating(ratings, movies, args.result_count,
                             args.csv_out, output)

        elif args.search_by == WATCHES_SB:
            # List movies by highest number of watches
            movies_by_watches(
                ratings, movies, args.result_count, args.csv_out, output)


# ===================================
# ======== FIND USERS BY IDS ========
# ===================================
def users_by_ids(ratings, movies, ids, csv_out, output):
    # Get ratings associated with users
    user_ratings = ratings.filter(ratings.userId.isin(ids))

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

    # Create user dataframe
    users = spark.createDataFrame(ids, "string").toDF("userId")

    # Join aggreations to user dataframe, removing duplicate userId columns
    users = users.join(user_ratings_agg, users.userId ==
                       user_ratings_agg.userId, "left").drop(user_ratings_agg.userId)

    users = users.join(user_movies_agg, users.userId ==
                       user_movies_agg.userId, "left").drop(user_movies_agg.userId)

    # Replace null values with 0
    users = users.na.fill(0, ["moviesWatched", "genresWatched"])

    if csv_out:
        output_csv(users, csv_out)
    else:
        # Display dataframe to output
        with redirect_stdout(output):
            users.show(len(ids), truncate=False)


# ===================================
# ======= FIND MOVIES BY IDS ========
# ===================================
def movies_by_ids(ratings, movies, ids, csv_out, output):
    # Find movies matching given IDs
    filtered_movies = movies.where(movies.movieId.isin(ids))

    # Print to output
    output_movies(ratings, filtered_movies, len(ids), csv_out, output)


# ===================================
# ====== FIND MOVIES BY TITLES ======
# ===================================
def movies_by_names(ratings, movies, titles, csv_out, output):
    # Find movies matching given titles
    filtered_movies = movies.where(movies.title.rlike("|".join(titles)))

    # Print to output
    output_movies(ratings, filtered_movies, len(titles), csv_out, output)


# ===================================
# ===== FIND MOVIES BY USER IDS =====
# ===================================
def movies_by_user_ids(ratings, movies, user_ids, out_count, csv_out, output):
    # Find ratings made by given users
    filtered_ratings = ratings.where(ratings.userId.isin(
        user_ids))

    # Find distinct movie IDs for each user via found ratings
    users_movies = filtered_ratings.select(
        col("userId"), col("movieId")).distinct()

    for user_id in user_ids:
        output.write("Movies for user ID " + user_id + ":\n")

        # Filter found movie IDs to this user only
        this_user_movie_ids = users_movies.where(
            users_movies.userId == user_id).select("movieId")

        # Join to movie dataframe to get full movie details
        this_user_movies = this_user_movie_ids.join(movies, "movieId")

        # Print movies
        output_movies(ratings, this_user_movies, out_count, csv_out, output)
        output.write("\n")


# ===================================
# ====== FIND MOVIES BY GENRES ======
# ===================================
def movies_by_genres(ratings, movies, genres, out_count, csv_out, output):
    for genre in genres:
        output.write(genre + ":\n")

        # Filter movies to genre
        genre_movies = movies.filter(array_contains(
            movies.genres, str(genre)))

        # Print movies
        output_movies(ratings, genre_movies, out_count, csv_out, output)
        output.write("\n")


# ===================================
# ======= FIND MOVIES BY YEAR =======
# ===================================
def movies_by_years(ratings, movies, years, out_count, csv_out, output):
    # Filter movies to years
    year_movies = movies.filter(movies.year.isin(years))

    # Print movies
    output_movies(ratings, year_movies, out_count, csv_out, output)
    output.write("\n")


# ===================================
# ====== LIST MOVIES BY RATING ======
# ===================================
def movies_by_rating(ratings, movies, out_count, output):
    # Find aggregate movie data (count & average)
    movies_agg = aggregate_movies(ratings, movies)

    # Order by descending average rating
    movies_agg = movies_agg.sort(
        col("avgRating").desc(), col("ratings").desc())

    if csv_out:
        output_csv(movies_agg, output)
    else:
        # Print to output
        with redirect_stdout(output):
            movies_agg.show(out_count, truncate=False)


# ===================================
# ====== LIST MOVIES BY WATCHES =====
# ===================================
def movies_by_watches(ratings, movies, out_count, csv_out, output):
    # Find aggregate movie data (count & average)
    movies_agg = aggregate_movies(ratings, movies)

    # Order by descending view count
    movies_agg = movies_agg.sort(
        col("ratings").desc(), col("avgRating").desc())

    if csv_out:
        output_csv(movies_agg, output)
    else:
        # Print to output
        with redirect_stdout(output):
            movies_agg.show(out_count, truncate=False)


# ===================================
# ====== FIND FAVOURITE GENRE =======
# ===================================
def find_favourite_genre(ratings, movies, user_ids, out_count, csv_out, output):
    # Find ratings made by given user
    filtered_ratings = ratings.where(ratings.userId.isin(user_ids))

    # Find associated movies
    movies_ratings = filtered_ratings.join(movies, "movieId")

    # Explode genre lists
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

    # Sort df by user ID & score
    scores = scores.sort(col("userId").asc(), col("score").desc())

    if csv_out:
        # Output CSV of found results
        output_csv(scores, output)

    # Round scores to 2 d.p.
    scores = scores.withColumn("score", spark_round(scores.score, 2))

    for user_id in user_ids:
        # Filter found scores to this user
        user_scores = scores.where(scores.userId == user_id)

        with redirect_stdout(output):
            print("User " + user_id + "'s favourite genre: " +
                  user_scores.first().genres)

            print("\nUser " + user_id + "'s scores:")
            user_scores.show(
                out_count, truncate=False)
            print("\n")


# ===================================
# ==== AGGREGATE MOVIES DATAFRAME ===
# ===================================
def aggregate_movies(ratings, movies):
    # Find ratings for movies
    movies_ratings = ratings.join(
        movies, ratings.movieId == movies.movieId, "inner").select(ratings["*"])

    # Aggregate ratings (count & average)
    rating_aggs = movies_ratings.groupBy(
        "movieId").agg(count("rating"), mean("rating"))

    # Rename columns
    rating_aggs = rating_aggs.withColumnRenamed(
        "count(rating)", "ratings").withColumnRenamed(
            "avg(rating)", "avgRating").withColumnRenamed(
                "movieId", "r_movieId")

    # Join to movies on movieId & remove duplicate ID column
    movies_agg = movies.join(rating_aggs, movies.movieId ==
                             rating_aggs.r_movieId, "left").drop("r_movieId")

    # Replace nulls with 0
    movies_agg = movies_agg.na.fill(0, ["ratings", "avgRating"])

    # Return aggregated data
    return movies_agg


# ===================================
# ====== PRINT MOVIES DATAFRAME =====
# ===================================
def output_movies(ratings, movies, out_count, csv_out, output):
    # Find aggregate data (count & average)
    movies_agg = aggregate_movies(ratings, movies)

    if csv_out:
        # Display dataframe to csv output
        output_csv(movies_agg, csv_out)
    else:
        # Display dataframe to output
        with redirect_stdout(output):
            movies_agg.show(out_count, truncate=False)


# ===================================
# ===== OUTPUT DATAFRAME AS CSV =====
# ===================================
def output_csv(df, csv_out):
    df.write.format("csv").mode("overwrite").option(
        "header", True).save(csv_out)


# ===================================
# ============ NAME GUARD ===========
# ===================================
if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Configure Spark Job
    spark = SparkSession.builder.master(
        "local").appName(APP_NAME).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Execute the program
    main(spark, args)
