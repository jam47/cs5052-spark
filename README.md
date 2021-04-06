# CS5052 P1 - Spark Data Analysis Application

## TODO

- Test all methods
  - Generate examples while at this
- Clean up all TODOs
- Number of users in each cluster
- Check all values for arguments are described
- Check if given values appear (otherwise we'll get null pointers)
      - E.g. searching for User ID which doesn't exist

- Part 3
- user-cluster.txt
- auto-user-cluster.txt
- movie-suggestions.txt

- Complete this readme

## Arg Validation required
- Reject -k if not clustering
- Reject -L if comparing users OR clustering
- Value cannot be specified for list top ratings/watches
    - Value must be specified for everything else




## Usage
----
The application should be executed with `Python3` using the following command.

```
p1.py -f <search_for_type> -b <search_by_type>
```

The options available for `<search_for_type>` and their uses are as follows.
- `users` : Search for users
- `movies` : Search for movies
- `favourite-genres` : Search for users' favourite genres
- `compare-users` : Compare users' preferences
- `cluster-users` : Cluster users based on their preferences
- `movie-suggestions` : Search for movies suggestions based on nearest neighbours

The options available for `<search_by_type>` and their uses are as follows.
- `user-ids` : Search by user IDs (must supply value using -v)
- `movie-ids` : Search by movie IDs (must supply value using -v)
- `movie-names` : Search by movie names (must supply value using -v)
- `genres` : Search by genre names (must supply value using -v)
- `rating` : List the top rated movies
- `watches` : List the most watched movies

**Note**: All search-for types, except for "movies", must use "user-ids" as their search-by type.

The optional arguments available and their uses are as follows.
- `-v, --value` : (Values) Provide the search value(s) for the search
- `-c, --count` : (Value) Select the number of search results to display
- `-h, --help` : (Flag) Display the help message and exit
- `-k, --num-centroids` : (Value) Select the number of centroids to be used for k-means clustering (min of 2)
- `-a, --auto-k-means` : (Flag) Use an automatically determined number of centroids for k-means clustering. Use "-k" to specify the maximum number of centroids.
- `-i, --input` : (Value) Provide the directory path for the data to be read
- `-o, --output` : (Value) Provide the output file for the program to write to
- `-L, --list` : (Value) Provide a filepath for the results to be outputted to as a CSV file

## Usage Examples
----

`p1.py -f users -b user-ids -v 1 2` : Searches for users 1 and 2 by ID.

`p1.py -f movies -b movie-names -c 30 -v cool` : Searches for movies with titles similar to \say{cool}, outputting 30 results.

`p1.py -f movies -b watches -c 50 -o example.txt` : Lists the top 50 most watched movies, outputting the results to \ttt{example.txt}.

`p1.py -f compare-users -b user-ids -v 5 6 7` : Compare the preferences of users with IDs 5, 6, and 7.

`p1.py -f cluster-users -b user-ids -a -v 10 11 12` : Train a k-means model using an automatically determined value of k and determine the clusters for users with IDs 10, 11, and 12.