from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark.sql.functions import col, substring, expr, split, explode, mean, count, array_contains, countDistinct
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import round as spark_round
import argparse
import sys
from datetime import datetime
from contextlib import redirect_stdout
from searches.user_searches import users_by_ids, formatted_user_genre_scores, user_taste_comparison
from searches.movie_searches import movies_by_genres, movies_by_titles, movies_by_ids, movies_by_user_ids, movies_by_years
from searches.movie_searches import movies_sorted_rating, movies_sorted_watches
from searches.cluster_searches import user_cluster_model, user_cluster_model_auto_k, get_cluster_model_centroids, get_cluster_model_silhouette, get_users_cluster_predictions


# ===================================
# ======== GLOBAL CONSTANTS =========
# ===================================

# Environment variables
SPARK_NAME = "CS5052 P1 - Apache Spark"             # Name of the spark app
APP_NAME = "170002815 & 170001567 - CS5052 P1"      # Name of the program
DEFAULT_DATASET_FILEPATH = "data/ml-latest/"        # Location of data directory
# Location of data directory
# DEFAULT_DATASET_FILEPATH = "data/ml-latest-small/"


# Dataset constants
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Search-for keys
USERS_SF = "users"
MOVIES_SF = "movies"
FAV_GENRES_SF = "favourite-genres"
COMPARE_USERS_SF = "compare-users"
CLUSTER_USERS_SF = "cluster-users"

# Search-by keys
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
                        choices=[USERS_SF, MOVIES_SF,
                                 FAV_GENRES_SF, COMPARE_USERS_SF, CLUSTER_USERS_SF],
                        help=("The type of entity to search for:"
                              "\n\t- " + USERS_SF + ": Search for users"
                              "\n\t- " + MOVIES_SF + ": Search for movies"
                              "\n\t- " + FAV_GENRES_SF + ": Search for users' favourite genres"
                              "\n\t- " + COMPARE_USERS_SF + ": Compare users' preferences"
                              "\n\t- " + CLUSTER_USERS_SF + ": Cluster users based on their preferences"
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
                        help="The value(s) to search by")

    # Results lengths
    parser.add_argument("-c", "--count", action="store", dest="out_count", default=10, type=int,
                        help="The number of results to return")

    # Results as csv list
    parser.add_argument("-l", "--list", dest="csv_out",
                        help="Output result as CSV to given filepath")

    # k for k-means clustering
    parser.add_argument("-k", "--num-centroids", dest="k", default=3, type=int,
                        help="The number of centroids to be used for k-means clustering (minimum of 2)")

    # Auto kmeans flag
    parser.add_argument("-a", "--auto-k-means", dest="auto_k_means", action="store_true",
                        help=("Automatically determine the values of k. "
                              "Use the '-k' argument to specify the maximum."))

    # Input directory
    parser.add_argument("-i", "--input", action="store", dest="input_dirpath",
                        default=DEFAULT_DATASET_FILEPATH,
                        help="The relative directorypath for the data to be read.")

    # Output file
    parser.add_argument("-o", "--output", action="store", dest="outfile",
                        help=("The relative filepath for the program's output to be written to. "
                              "If no file value is supplied, output will be written to stdout."))

    # Parse args
    args = parser.parse_args()

    # Validate for-by search combinations
    if ((args.search_for == USERS_SF
         or args.search_for == FAV_GENRES_SF
         or args.search_for == COMPARE_USERS_SF
         or args.search_for == CLUSTER_USERS_SF)
            and args.search_by != USERS_SB):

        print("Invalid Arguments: Invalid search combination\n")
        parser.print_help()
        sys.exit(0)

    # Enforce search value for certain search-bys
    if ((args.search_by == USERS_SB
         or args.search_by == MOVIE_IDS_SB
         or args.search_by == MOVIE_NAMES_SB
         or args.search_by == YEARS_SB
         or args.search_by == GENRES_SB)
            and len(args.search_value) == 0):

        print("Invalid Arguments: At least one search value (-v) must be provided")
        parser.print_help()
        sys.exit(0)

    # Other argument validation
    if args.k <= 1:
        print("Invalid Arguments: k (-k) must be 2 or more for k-means clustering\n")
        parser.print_help()
        sys.exit(0)

    if args.out_count <= 0:
        print("Invalid Arguments: count (-c) must be positive\n")
        parser.print_help()
        sys.exit(0)

    return args


# ===================================
# ====== AUXILIARY FUNCTIONS  =======
# ===================================

def df_to_output(dataframe, out_count, output):
    """ Outputs a out_count rows of dataframe to output """

    with redirect_stdout(output):
        dataframe.show(out_count, truncate=False)


def df_to_csv(dataframe, csv_out):
    """ Outputs dataframe in CSV format to csv_out """

    dataframe.write.format("csv").mode("overwrite").option(
        "header", True).save(csv_out)


def output_dataframe(dataframe, output, out_count, csv_out):
    """ Outputs a given dataframe using both df_to_output and df_to_csv """

    df_to_output(dataframe, out_count, output)

    if args.csv_out is not None:
        df_to_csv(dataframe, csv_out)


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
        "Program: " + APP_NAME +
        "\nCommand: " + " ".join(sys.argv) +
        "\nData directory: " + args.input_dirpath +
        "\nDate: " + datetime.now().strftime("%Y.%m.%d %H:%M") +
        "\n\n"
    ))

    # Read dataset
    ratings = spark.read.csv(args.input_dirpath + "/ratings.csv", header=True)
    movies = spark.read.csv(args.input_dirpath + "/movies.csv", header=True)

    ratings = ratings.alias('ratings')
    movies = movies.alias('movies')

    # Split genres string into array
    movies = movies.withColumn("genres", split(
        "genres", "\|"))

    # Separate years from movie titles
    movies = movies.withColumn("year", substring(col("title"), -5, 4))
    movies = movies.withColumn("title", expr(
        "substring(title, 1, length(title)-7)"))

    # Cast numerical columns
    ratings = ratings\
        .withColumn("rating", ratings.rating.cast(types.FloatType()))\
        .withColumn("userId", ratings.userId.cast(types.IntegerType()))

    # Cache dataframes to avoid reloading
    ratings.cache()
    movies.cache()

    # Perform search
    if args.search_for == USERS_SF:
        if args.search_by == USERS_SB:
            # Search for users by IDs & display all
            result = users_by_ids(spark, ratings, movies, args.search_value)
            output_dataframe(result, output, args.out_count, args.csv_out)

    elif args.search_for == FAV_GENRES_SF:
        if args.search_by == USERS_SB:
            # Get users' genre scores
            scores = formatted_user_genre_scores(
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
                df_to_output(user_scores, args.out_count, output)
                output.write("\n")

    elif args.search_for == COMPARE_USERS_SF:
        # Generate comparison between given users
        to_output = user_taste_comparison(
            spark, ratings, movies, args.search_value)
        output.write(to_output)

    elif args.search_for == CLUSTER_USERS_SF:
        if args.auto_k_means:
            # Generate clustering model using optimal k centroids
            (cluster_model, best_k, score_dict) = user_cluster_model_auto_k(
                spark, ratings, movies, args.k, GENRES)

            # Display calculated scores
            output.write("Clustering models silhouette scores: ")
            for k in score_dict:
                output.write("\n  - k=" + str(k) + ": " + str(score_dict[k]))
                if k == best_k:
                    output.write(" (Optimal)")

        else:
            # Generate clustering model using fixed k centroids
            cluster_model = user_cluster_model(
                spark, ratings, movies, args.k, GENRES)

            # Output model's silhouette score
            to_output = get_cluster_model_silhouette(cluster_model)
            output.write(
                "Clustering model silhouette score: " + str(to_output))

        # Output model's centroids
        to_output = get_cluster_model_centroids(cluster_model)
        output.write("\n\nClustering model centroids: " + to_output)

        # Make & output predictions for given users
        predictions_df = get_users_cluster_predictions(
            spark, ratings, movies, cluster_model, args.search_value)

        output.write("\n\nCluster predictions for users " +
                     ", ".join(args.search_value) + ":\n")

        output_dataframe(predictions_df, output, len(
            args.search_value), args.csv_out)

    elif args.search_for == MOVIES_SF:
        if args.search_by == MOVIE_IDS_SB:
            # Search for movies by IDs & display
            result = movies_by_ids(
                spark, ratings, movies, args.search_value)
            output_dataframe(result, output, args.out_count, args.csv_out)

        elif args.search_by == MOVIE_NAMES_SB:
            # Search for movies by names & display
            result = movies_by_titles(
                spark, ratings, movies, args.search_value)
            output_dataframe(result, output, args.out_count, args.csv_out)

        elif args.search_by == GENRES_SB:
            # Search for movies by genres
            genres_movies = movies_by_genres(
                spark, ratings, movies, args.search_value)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(genres_movies, args.csv_out)

            for genre in args.search_value:
                # Filter found movies to this genre
                this_genre_movies = genres_movies.where(
                    array_contains(genres_movies.genres, genre))

                # Display results
                output.write("\nMovies in " + genre + " genre:\n")
                df_to_output(this_genre_movies, args.out_count, output)

        elif args.search_by == USERS_SB:
            # Search for movies by user IDs
            users_movies = movies_by_user_ids(spark, ratings, movies,
                                              args.search_value)

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
                df_to_output(this_user_movies, args.out_count, output)
                output.write("\n")

        elif args.search_by == YEARS_SB:
            # Search for movies by given years
            movies_in_years = movies_by_years(
                spark, ratings, movies, args.search_value)

            if args.csv_out is not None:
                # Print to CSV if requested
                df_to_csv(movies_in_years, args.csv_out)

            for year in args.search_value:
                # Filter found movies to this year
                this_year_movies = movies_in_years.where(
                    movies_in_years.year == year)

                # Display
                output.write("Movies in " + year + ":\n")
                df_to_output(this_year_movies, output,
                             args.out_count, args.csv_out)

        elif args.search_by == RATING_SB:
            # Get movies sorted by average rating & display
            result = movies_sorted_rating(spark, ratings, movies)
            output_dataframe(result, output, args.out_count, args.csv_out)

        elif args.search_by == WATCHES_SB:
            # Get movies sorted by number of ratings & display
            result = movies_sorted_watches(spark, ratings, movies)
            output_dataframe(result, output, args.out_count, args.csv_out)

    output.close()
    return


# ===================================
# ============ NAME GUARD ===========
# ===================================
if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Configure Spark Job
    spark = SparkSession.builder.master("local")\
        .appName(SPARK_NAME).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Execute program
    main(spark, args)
