# Thesis blog

A bunch of (mostly) rendered Jupyter notebooks about my thesis work. The blog is hosted on GitHub Pages and is available at [jindrich.bar/thesis-blog](https://jindrich.bar/edu/thesis-blog).

While I try to make the posts make sense on their own, it's possible they will be hard to understand without the context at times.
To learn more about the motivation behind the problems solved here, read the [thesis](https://barjin.github.io/master-thesis/bar-social-network-analysis-in-academic-environment-2024.pdf) first and treat these posts as the supportive "behind the scenes" material.

## Posts

### [10.5.2024 | Ranking benchmarks](./ranking-benchmarks/index.md)

Part of the thesis work is to evaluate the performance of the basic ranking algorithm - and see whether we can improve it by utilizing the graph structure of the data. 
This post is about creating the benchmark we will use to evaluate the performance of the algorithm.

### [27.5.2024 | Collecting the gold-standard benchmarking data](./collecting-data/index.md)

A short post about collecting, cleaning and exploring the Elsevier Scopus data based on the queries from the previous post.

### [29.5.2024 | Benchmarking NDCG scores](./ndcg-benchmark/index.md)

With the collected Scopus and Charles Explorer data, in this post we are exploring the possibilities of the automated relevance feedback system based on LLM-embeddings based similarity search and we see whether (and how much) do the measures of the local graph structure improve the search result ranking.

### [7.7.2024 | Hierarchical merging - distance matrix calculation](./inference-distance-matrix/index.md)

The data exports from the university systems are missing detailed data about external authors from other universities. 
One large problem is that the external authors are only identified by their name.
To automatically merge the graph nodes representing the same person, we try to employ a *hierarchical merging* algorithm. 
In this blog post, we benchmark different ways of calculating the distance matrix between the mergeable nodes.
