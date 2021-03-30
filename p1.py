from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark.sql.functions import col, substring, expr, split, explode, mean, count, array_contains, countDistinct
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import round as spark_round
import argparse
import sys
from datetime import date
from contextlib import redirect_stdout
# from user_searches
# import movie_searches
from searches.user_searches import users_by_ids, user_genre_scores
from searches.movie_searches import movies_by_genres, movies_by_titles, movies_by_ids, movies_by_user_ids, movies_by_years
from searches.movie_searches import movies_sorted_rating, movies_sorted_watches


# ===================================
# ======== GLOBAL CONSTANTS =========
# ===================================

# Env variables
APP_NAME = "CS5052 P1 - Apache Spark"
DATASET_FILEPATH = "data/ml-latest-small-modified/"

# Search for keys
USERS_SF = "users"
MOVIES_SF = "movies"
FAV_GENRES_SF = "favourite-genres"
COMPARE_USERS_SF = "compare-users"

# Search by keys
USERS_SB = "user-ids"
MOVIE_IDS_SB = "movie-ids"
MOVIE_NAMES_SB = "movie-names"
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
                              "\n\t- " + FAV_GENRES_SF + ": Search for users' favourite genres"
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

    if args.search_for == FAV_GENRES_SF and args.search_by != USERS_SB:
        parser.print_help()
        sys.exit(0)

    # TODO - Enforce value for certain searches

    return args


# ===================================
# ====== AUXILIARY FUNCTIONS  =======
# ===================================

def output_dataframe(dataframe, out_count, output):
    """ Outputs a out_count rows of dataframe to output """

    with redirect_stdout(output):
        dataframe.show(out_count, truncate=False)


def df_to_csv(df, csv_out):
    """ Outputs a given dataframe (df) in CSV format to csv_out """

    df.write.format("csv").mode("overwrite").option(
        "header", True).save(csv_out)


# ===================================
# ============ NAME GUARD ===========
# ===================================

if __name__ == "__main__":
    # Configure Spark Job
    spark = SparkSession.builder.master(
        "local").appName(APP_NAME).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Parse arguments
    args = parse_args()

    # Execute program
    main(spark, args)


# ===================================
# =========== MAIN METHOD ===========
# ===================================

def main(spark, args):

    # Open outfile
    if args.outfile is not None and args.outfile != "":
        output = open(args.outfile, "w")
    else:
        output = sys.stdout

    # Print log header & given command
    output.write((
        "Program: 170002815 & 170001567's CS5052 Submission"
        "\nDate: " + date.today().strftime("%d.%m.%Y") +
        "\nCommand: " + " ".join(sys.argv) + "\n\n"
    ))

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
            # Search for users by IDs & display all
            users = users_by_ids(spark, ratings, movies, args.search_value)
            output_dataframe(users, users.count(), output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(users, args.csv_out)

    elif args.search_for == FAV_GENRES_SF:
        if args.search_by == USERS_SB:
            # Get users' genre scores
            scores = user_genre_scores(
                spark, ratings, movies, args.search_value)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(scores, args.csv_out)

            for user_id in args.search_value:
                # Filter found scores to this user
                user_scores = scores.where(scores.userId == user_id)

                # Output user's highest score (i.e. favourite)
                output.write("User " + user_id + "'s favourite genre: " +
                             user_scores.first().genres)

                # Output user's scores
                output.write("\nUser " + user_id + "'s scores:")
                output_dataframe(user_scores, )
                output.write("\n")

    elif args.search_for == MOVIES_SF:
        if args.search_by == MOVIE_IDS_SB:
            # Search for movies by IDs & display
            result = movies_by_ids(
                spark, ratings, movies, args.search_value)
            output_dataframe(result, args.out_count, output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(result, args.csv_out)

        elif args.search_by == MOVIE_NAMES_SB:
            # Search for movies by names & display
            result = movies_by_titles(
                spark, ratings, movies, args.search_value)
            output_dataframe(result, args.out_count, output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(result, args.csv_out)

        elif args.search_by == GENRES_SB:
            # Search for movies by genres & display
            result = movies_by_genres(
                spark, ratings, movies, args.search_value)
            output_dataframe(result, args.out_count, output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(result, args.csv_out)

        elif args.search_by == USERS_SB:
            # Search for movies by user IDs
            users_movies = movies_by_user_ids(
                ratings, movies, args.search_value)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(users_movies, args.csv_out)

            for user_id in args.search_value:
                # Filter found movie IDs to this user
                this_user_movie_ids = users_movies.where(
                    users_movies.userId == user_id).select("movieId")

                # Join to movie dataframe to get full movie details
                this_user_movies = this_user_movie_ids.join(movies, "movieId")

                # Display
                output.write("Movies for user ID " + user_id + ":\n")
                output_dataframe(this_user_movies, args.out_count, output)
                output.write("\n")

        elif args.search_by == YEARS_SB:
            # Search for movies by year & display
            result = movies_by_years(spark, ratings, movies, args.search_value)
            output_dataframe(result, args.out_count, output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(result, args.csv_out)

        elif args.search_by == RATING_SB:
            # Get movies sorted by average rating & display
            result = movies_sorted_rating(spark, ratings, movies)
            output_dataframe(result, args.out_count, output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(result, args.csv_out)

        elif args.search_by == WATCHES_SB:
            # Get movies sorted by number of ratings & display
            result = movies_sorted_watches(spark, ratings, movies)
            output_dataframe(result, args.out_count, output)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(result, args.csv_out)
