# Thesis blog

A bunch of (mostly) rendered Jupyter notebooks about my thesis work. The blog is hosted on GitHub Pages and is available at [jindrich.bar/thesis-blog](https://jindrich.bar/edu/thesis-blog).

## Posts

### [10.5.2024 | Ranking benchmarks](./ranking-benchmarks/index.md)

Part of the thesis work is to evaluate the performance of the basic ranking algorithm - and see whether we can improve it by utilizing the graph structure of the data. 
This post is about creating the benchmark we will use to evaluate the performance of the algorithm.

### [27.5.2024 | Collecting the gold-standard benchmarking data](./collecting-data/index.md)

A short post about collecting, cleaning and exploring the Elsevier Scopus data based on the queries from the previous post.

### [29.5.2024 | Benchmarking NDCG scores](./ndcg-benchmark/index.md)

With the collected Scopus and Charles Explorer data, in this post we are exploring the possibilities of the automated relevance feedback system based on LLM-embeddings based similarity search and we see whether (and how much) do the measures of the local graph structure improve the search result ranking.
