# cs5052-spark

## TODO

- Swap function arguments with `args` variable
- Split functions into `user`, `movies` and `util` file
  - Could also maybe split by task in the spec?
- Re-add csv output option properly
  - ouput to `stdout` by default, optionally include `csv` output with pyspark function
- Print executed command to output

## NOTES FOR PART 2.2

IDEAS:

- Normalise users' scores somehow (min/max or standardisation)
- Standardisation probs better because we can set unrated to 0
- Could find most watched movie (normalised ofc)

1. Sum and normalise views
2. Add corresponding films
3. Order by largest
4. Take top 5

5. Sum and normalise ratings
6. calculate differences
7. show largest and smallest differences
8. threshold 0.5 avg difference as 'similar taste' (and other fixed thresholds)
