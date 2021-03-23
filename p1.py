from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, mean
import sys
import argparse

# Env variables
APP_NAME = "CS5052 P1 - Apache Spark"
DATASET_FILEPATH = "data/ml-latest-small/"

# Search for keys
USERS_SF = "users"
MOVIES_SF = "movies"

# Search by keys
USERS_SB = "users"
MOVIE_IDS_SB = "movie_ids"
MOVIE_NAMES_SB = "movie_names"
YEAR_SB = "year"
GENRES_SB = "genres"
RATING_SB = "rating"
WATCHES_SB = "watches"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Search for
    parser.add_argument("-f", "--search-for", action="store", dest="search_for",
                        choices=[USERS_SF, MOVIES_SF],
                        help=("The type of entity to search for:"
                              "\n\t- " + USERS_SF + ": Search for users"
                              "\n\t- " + MOVIES_SF + ": Search for movies"
                              ))

    # Search by
    parser.add_argument("-b", "--search-by", action="store", dest="search_by",
                        choices=[USERS_SB, MOVIE_IDS_SB, MOVIE_NAMES_SB,
                                 YEAR_SB, RATING_SB, WATCHES_SB],
                        help=("The type of entity to search by:"
                              "\n\t- " + USERS_SB + " : Search by user IDs"
                              "\n\t- " + MOVIE_IDS_SB + " : Search by movie IDs"
                              "\n\t- " + MOVIE_NAMES_SB + " : Search by movie names"
                              "\n\t- " + GENRES_SB + " : Search by genre names"
                              "\n\t- " + RATING_SB + " : List top rated movies"
                              "\n\t- " + WATCHES_SB + " : List most watched movies"
                              ))

    # Search value
    parser.add_argument("-v", "--value", action="store", dest="search_value", nargs='+',  # One or more arguments needs to be last argument
                        help="The value to search by")

    # Results length
    parser.add_argument("-c", "--count", action="store",
                        dest="result_count", help="The number of results to return")

    # Output file
    parser.add_argument("-o", "--output", action="store", dest="out",
                        help="The output filepath for the program. If no file value is supplied, output will be written to stdout.")

    # Parse args & validate for-by search combination
    args = parser.parse_args()

    if args.search_for == USERS_SF and args.search_by != USERS_SB:
        parser.print_help()
        sys.exit(0)

    if args.search_for == MOVIES_SF and args.search_by == USERS_SB:
        parser.print_help()
        sys.exit(0)

    return args


def main(spark, args):
    # Open outfile
    if args.out is not None and args.out != "":
        output = open(args.out, "w")
    else:
        output = sys.stdout
    # TODO - Use this output

    # Read dataset
    ratings = spark.read.csv(DATASET_FILEPATH + "/ratings.csv", header=True)
    movies = spark.read.csv(DATASET_FILEPATH + "/movies.csv", header=True)

    ratings = ratings.alias('ratings')
    movies = movies.alias('movies')

    # Split geres into array
    movies = movies.withColumn("genres", split(
        "genres", "\|"))

    # Persist dataset
    ratings.cache()
    movies.cache()
    # TODO - Cache others here

    # display number of movies/genre for particular user

    if args.search_for == USERS_SF:
        if args.search_by == USERS_SB:
            userById(ratings, movies, args.search_value, output)
    elif args.search_for == MOVIES_SF:
        if args.search_by == MOVIE_IDS_SB:
            movieById(ratings, movies, args.search_value, output)
        elif args.search_by == MOVIE_NAMES_SB:
            movieByName(ratings, movies, args.search_value,
                        output)  # TODO - Not Implemented


# Find users by user ids
def userById(ratings, movies, search_value, output):
    output.write("userId,moviesWatched,genres\n")
    for id in search_value:
        output.write(str(id)+",")
        user_ratings = ratings.where(ratings.userId == id)
        output.write(str(user_ratings.count())+",")
        user_movies = movies.join(
            user_ratings, user_ratings.movieId == movies.movieId, "inner").select(movies["*"])
        user_genres = user_movies.select(explode(user_movies.genres))
        output.write(str(user_genres.distinct().count())+"\n")

# Find movies by ids
def movieById(ratings, movies, search_value, output):
    output.write("movieId,title,avgRating,views\n")
    for id in search_value:
        movie = movies.where(movies.movieId == id)
        outputMovie(ratings, movie, output)

# Find movies by titles
def movieByName(ratings, movies, search_value, output):
    output.write("movieId,title,avgRating,views\n")
    for title in search_value:
        movie = movies.where(movies.title == title)
        outputMovie(ratings, movie, output)


def outputMovie(ratings, movie, output):
    output.write(str(movie.first().movieId)+",")
    output.write(str(movie.first().title)+",")
    movie_ratings = ratings.join(
        movie, ratings.movieId == movie.movieId, "inner").select(ratings["*"])
    output.write(str(movie_ratings.select(
        mean(movie_ratings.rating)).first()['avg(rating)'])+",")
    output.write(str(movie_ratings.count())+"\n")


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Configure Spark Job
    spark = SparkSession.builder.master(
        "local").appName(APP_NAME).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Execute the program
    main(spark, args)
