<( [Collecting the gold-standard data for benchmarking](../collecting-data/index.md) )

# Measuring the current search result ranking

> This Python notebook shows the process of benchmarking the search result ranking for the Charles Explorer application.
> It is a part of my diploma thesis at the Faculty of Mathematics and Physics, Charles University, Prague.
>
> Find out more about the thesis in the [GitHub repository](https://github.com/barjin/master-thesis).
>
> Made by Jindřich Bär, 2024. 

In the previous posts, we have established a methodology for benchmarking the different search result ranking algorithms. We have also collected the relevance scores for the benchmark, since the Charles Explorer application does not provide enough usage data to use for the benchmarking yet. 

In this post, we will measure the current search result ranking of the Charles Explorer application to establish the baseline for the benchmarking.

We start by comparing the results from both search engines (Charles Explorer and Elsevier Scopus). 


```python
import pandas as pd

queries = pd.read_csv('./best_queries.csv')
scopus = pd.read_csv('./scopus_results.csv')
explorer = pd.read_csv('./filtered_search_results.csv')

original_queries = set(queries['0'].to_list())
scopus_queries = set(scopus['query'].unique())
print(f"Scopus missing queries: {len(original_queries.difference(scopus_queries))}")
```

    Scopus missing queries: 25


For 25 queries out of the original query set of 174 queries, Scopus does not return any results. We will exclude these queries from the benchmarking.

We proceed by calculating the prceision, recall and F1 score for the search results of the Charles Explorer application - considering the Scopus search results as the ground truth.


```python
from utils.calc.precision_recall import get_precision, get_recall

queries = scopus['query'].unique()

precision = pd.Series(queries).map(
    lambda query: get_precision(
        explorer[explorer['query'] == query]['name'].to_list(), 
        scopus[scopus['query'] == query]['title'].to_list(), 
        lambda x: x.lower()
    )
)

recall = pd.Series(queries).map(
    lambda query: get_recall(
        explorer[explorer['query'] == query]['name'].to_list(), 
        scopus[scopus['query'] == query]['title'].to_list(), 
        lambda x: x.lower()
    )
)

f1 = 2 * (precision * recall) / (precision + recall)


metrics = pd.DataFrame({
    'query': queries,
    'precision': precision,
    'recall': recall,
    'f1': f1
})

metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>query</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>physics</td>
      <td>0.043011</td>
      <td>0.040000</td>
      <td>0.041451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bolus</td>
      <td>0.125000</td>
      <td>0.121212</td>
      <td>0.123077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Plavix</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>draft</td>
      <td>0.010870</td>
      <td>0.010753</td>
      <td>0.010811</td>
    </tr>
    <tr>
      <th>4</th>
      <td>block</td>
      <td>0.144330</td>
      <td>0.141414</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>144</th>
      <td>metaphysics</td>
      <td>0.020833</td>
      <td>0.037736</td>
      <td>0.026846</td>
    </tr>
    <tr>
      <th>145</th>
      <td>angiogenesis inhibitor</td>
      <td>0.187500</td>
      <td>0.044118</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>146</th>
      <td>sports medicine</td>
      <td>0.109589</td>
      <td>0.163265</td>
      <td>0.131148</td>
    </tr>
    <tr>
      <th>147</th>
      <td>clinical neurology</td>
      <td>0.088889</td>
      <td>0.666667</td>
      <td>0.156863</td>
    </tr>
    <tr>
      <th>148</th>
      <td>specific</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 4 columns</p>
</div>




```python
metrics['f1'].describe()
```




    count    114.000000
    mean       0.208727
    std        0.211699
    min        0.010101
    25%        0.074786
    50%        0.137028
    75%        0.265263
    max        1.000000
    Name: f1, dtype: float64



These calculations show us that the mean `f1` score over all available queries is `0.21`. This shows that the current Charles Explorer search results differ quite a lot from the Scopus search results. This can be caused by mutiple reasons - either the publications are not present in the Scopus database, or the queries are not specific enough - and the search results are returning partially disjoint sets of publications.

This is an issue that cannot be solved by simply re-ranking the Charles Explorer search results. Changing the ordering of the search results will not help if the search results themselves are not relevant. Moreover, this also limits our ability to use the Scopus search results ranking as the proxy for the relevance feedback.

We are already trying to mitigate the issue of the broad queries by evaluating the benchmark on a larger-than-usual sets of results. By collecting data for the top `100` search results (Charles Explorer only shows the first 30 results by default), we could see if the missing publications are just ranked lower or if they are not present at all.

Since this thesis is focused on the search result **ranking** algorithms, we will proceed with the benchmarking as planned.
However, to improve the relevance score assignment, we try adding a *similarity search step*. 
Currently, we are only matching the Charles Explorer search results with the Scopus search results by the publication title (case-insensitive). 
This matching criterion is prone to even the slightest variations in the publication titles, which can lead to false negatives.

## Infering the relevance with similarity search

In the proposed *similarity search step*, we use the similarity of LLM (Large Language Model) embeddings to match the publication titles. This should help us to relate the publications missing from the Scopus search results to the ones present there and assign them a relevance score.

The *similarity search step* works as follows:

1. By the means of a LLM embedding model, we calculate the embeddings for the publication titles of the Elsevier Scopus search results. We store these embeddings in a vector database.
3. For each publication title in the Charles Explorer search results, we calculate its embedding. In the database, we search for the nearest embedding among Scopus search results embeddings. Futhermore, we require the retrieved document to be a result of the same query (in Elsevier Scopus) as the Charles Explorer search result.
4. We calculate original document's relevance from the retrieved document's attributes - e.g. its position in Scopus search.

> The LLM embeddings are vector representations of the publication titles. While those can be arbitrary vectors, they are usually optimized to capture the **semantic meaning** of the text. This means that texts with similar meanings should have similar embeddings - i.e. the (cosine) similarity of the embedding vectors should be high.

---

For the purpose of this thesis, we are embedding the documents using the `all-MiniLM-L6-v2` sentence-transformer model. This model is a general-purpose English embedding model designed for running on consumer-grade hardware. Due to its small size and competitive performance, it's often used for the real-time use-cases, like semantic search or RAG (Retrieval-Augmented Generation).

https://www.sbert.net/docs/sentence_transformer/pretrained_models.html


```python
from utils.llm.get_vector_db import vectorstore, embedder
import time
import warnings

warnings.filterwarnings("ignore")

KNN = 1

def get_similarity_position(records: list[str], query: str):
    embedding_start = time.time()
    vectors = embedder.embed_documents(records)
    embedding_end = time.time()

    results = []

    for vector in vectors:
        vector_db_start = time.time()
        result = vectorstore.similarity_search_by_vector(vector, KNN, filter={'query': query})
        vector_db_end = time.time()

        if(len(result) == 0):
            results.append(None)
        else:
            results.append({
                'ranking': result[0].metadata.get('ranking'),
                'embedding_duration': (embedding_end - embedding_start) / len(records),
                'vector_db_roundtrip': vector_db_end - vector_db_start
            })

    return results


def get_positions():
    new_df = pd.DataFrame()
    done=0
    total=len(explorer)

    for query in explorer['query'].unique():
        records = explorer[explorer['query'] == query]
        records.reset_index(inplace=True)
        similarities = get_similarity_position(records['name'].to_list(), query)
        records['ranking'] = [similarity['ranking'] if similarity is not None else None for similarity in similarities]
        records['embedding_duration'] = [similarity['embedding_duration'] if similarity is not None else None for similarity in similarities]
        records['vector_db_roundtrip'] = [similarity['vector_db_roundtrip'] if similarity is not None else None for similarity in similarities]

        new_df = pd.concat([new_df, records])

        done += len(records)
        print(f"Done {done} out of {total}")

    return new_df

get_positions().drop(columns=['index']).to_csv('./similarity_ranking.csv', index=False)
```

    Done 64 out of 9269
    Done 130 out of 9269
    Done 230 out of 9269
    Done 269 out of 9269
    Done 341 out of 9269
    Done 441 out of 9269
    Done 442 out of 9269
    Done 487 out of 9269
    Done 587 out of 9269
    Done 596 out of 9269
    Done 597 out of 9269
    Done 613 out of 9269
    Done 638 out of 9269
    Done 723 out of 9269
    Done 823 out of 9269
    Done 923 out of 9269
    Done 924 out of 9269
    Done 927 out of 9269
    Done 929 out of 9269
    Done 930 out of 9269
    Done 1030 out of 9269
    Done 1031 out of 9269
    Done 1039 out of 9269
    Done 1139 out of 9269
    Done 1143 out of 9269
    Done 1156 out of 9269
    Done 1161 out of 9269
    Done 1203 out of 9269
    Done 1204 out of 9269
    Done 1205 out of 9269
    Done 1305 out of 9269
    Done 1307 out of 9269
    Done 1308 out of 9269
    Done 1408 out of 9269
    Done 1508 out of 9269
    Done 1521 out of 9269
    Done 1542 out of 9269
    Done 1587 out of 9269
    Done 1687 out of 9269
    Done 1734 out of 9269
    Done 1735 out of 9269
    Done 1736 out of 9269
    Done 1836 out of 9269
    Done 1879 out of 9269
    Done 1979 out of 9269
    Done 1981 out of 9269
    Done 1983 out of 9269
    Done 2083 out of 9269
    Done 2085 out of 9269
    Done 2104 out of 9269
    Done 2204 out of 9269
    Done 2253 out of 9269
    Done 2353 out of 9269
    Done 2365 out of 9269
    Done 2367 out of 9269
    Done 2418 out of 9269
    Done 2518 out of 9269
    Done 2519 out of 9269
    Done 2545 out of 9269
    Done 2645 out of 9269
    Done 2648 out of 9269
    Done 2748 out of 9269
    Done 2773 out of 9269
    Done 2873 out of 9269
    Done 2891 out of 9269
    Done 2991 out of 9269
    Done 3091 out of 9269
    Done 3105 out of 9269
    Done 3109 out of 9269
    Done 3209 out of 9269
    Done 3309 out of 9269
    Done 3311 out of 9269
    Done 3325 out of 9269
    Done 3336 out of 9269
    Done 3338 out of 9269
    Done 3429 out of 9269
    Done 3529 out of 9269
    Done 3531 out of 9269
    Done 3631 out of 9269
    Done 3731 out of 9269
    Done 3831 out of 9269
    Done 3931 out of 9269
    Done 4016 out of 9269
    Done 4048 out of 9269
    Done 4148 out of 9269
    Done 4149 out of 9269
    Done 4249 out of 9269
    Done 4309 out of 9269
    Done 4310 out of 9269
    Done 4410 out of 9269
    Done 4510 out of 9269
    Done 4546 out of 9269
    Done 4587 out of 9269
    Done 4588 out of 9269
    Done 4589 out of 9269
    Done 4689 out of 9269
    Done 4691 out of 9269
    Done 4786 out of 9269
    Done 4820 out of 9269
    Done 4831 out of 9269
    Done 4835 out of 9269
    Done 4935 out of 9269
    Done 4993 out of 9269
    Done 5093 out of 9269
    Done 5131 out of 9269
    Done 5231 out of 9269
    Done 5239 out of 9269
    Done 5339 out of 9269
    Done 5439 out of 9269
    Done 5539 out of 9269
    Done 5566 out of 9269
    Done 5580 out of 9269
    Done 5673 out of 9269
    Done 5689 out of 9269
    Done 5694 out of 9269
    Done 5755 out of 9269
    Done 5855 out of 9269
    Done 5881 out of 9269
    Done 5981 out of 9269
    Done 6069 out of 9269
    Done 6169 out of 9269
    Done 6174 out of 9269
    Done 6192 out of 9269
    Done 6292 out of 9269
    Done 6293 out of 9269
    Done 6393 out of 9269
    Done 6453 out of 9269
    Done 6482 out of 9269
    Done 6582 out of 9269
    Done 6583 out of 9269
    Done 6683 out of 9269
    Done 6692 out of 9269
    Done 6783 out of 9269
    Done 6827 out of 9269
    Done 6927 out of 9269
    Done 7027 out of 9269
    Done 7097 out of 9269
    Done 7179 out of 9269
    Done 7203 out of 9269
    Done 7303 out of 9269
    Done 7403 out of 9269
    Done 7503 out of 9269
    Done 7603 out of 9269
    Done 7690 out of 9269
    Done 7712 out of 9269
    Done 7812 out of 9269
    Done 7813 out of 9269
    Done 7833 out of 9269
    Done 7839 out of 9269
    Done 7840 out of 9269
    Done 7856 out of 9269
    Done 7860 out of 9269
    Done 7915 out of 9269
    Done 7920 out of 9269
    Done 7935 out of 9269
    Done 8035 out of 9269
    Done 8100 out of 9269
    Done 8200 out of 9269
    Done 8201 out of 9269
    Done 8301 out of 9269
    Done 8401 out of 9269
    Done 8501 out of 9269
    Done 8601 out of 9269
    Done 8701 out of 9269
    Done 8801 out of 9269
    Done 8803 out of 9269
    Done 8805 out of 9269
    Done 8874 out of 9269
    Done 8909 out of 9269
    Done 8985 out of 9269
    Done 8998 out of 9269
    Done 9098 out of 9269
    Done 9169 out of 9269
    Done 9269 out of 9269


This expands the table with the Charles Explorer search results with the rankings of the most similar publications from the Scopus database for the given queries.

As a reminder - we are trying to use these values as a proxy for the relevance of the search results. In a way, this replaces a manual relevance score assignment, which would require us to conduct a user survey to get the relevance scores for the search results.

We can now plot the distribution of the positions of the most similar publications in the Scopus search results for the Charles Explorer search results. This will help us to see whether the similarity search step helps with the missing publications problem.


```python
import pandas as pd

df = pd.read_csv('./similarity_ranking.csv')
df.rename(columns={'ranking': 'similarity_scopus_ranking'}, inplace=True)
c = df['similarity_scopus_ranking'].hist(bins=100)
c.set_title('Similarity-based ranking prediction for Charles Explorer documents')
c.set_xlabel('Predicted position in the Scopus search results')
c.set_ylabel('Number of documents')
```




    Text(0, 0.5, 'Number of documents')




    
![png](03_benchmarking_files/03_benchmarking_10_1.png)
    


While the distribution of the predicted rankings might seem skewed, this does not pose a serious problem to our benchmark.

Firstly, we are not trying to predict the exact ranking of the search results, but rather to assign a relevance score to each search result. The peak of the distribution is at the top of the rankings, which simulates the human behaviour of having clearer opinions about the few top search results.

Secondly, the left-skewed distribution might be caused by non-uniform lengths of the search result lists. Since for some of the queries, Scopus returns only a few relevant search results (100 is only the maximum limit), the resulting predicted rankings will be skewed towards the top of the list for these queries.


```python
h = scopus.groupby('query')['ranking'].max().hist(bins=100)

h.set_title('Histogram of lengths of the Scopus search results')
h.set_xlabel('Number of documents in the result list')
h.set_ylabel('Query count')
```




    Text(0, 0.5, 'Query count')




    
![png](03_benchmarking_files/03_benchmarking_12_1.png)
    


We now set the target value for the search ranking benchmark. We add the original positions of the search results as a separate column.


```python
rankings = pd.DataFrame()

for query in df['query'].unique():
    x = df[df['query'] == query].reset_index()
    x['charles_explorer_ranking'] = x.index
    rankings = pd.concat([
        rankings,
        x
    ])

rankings.drop(columns=['index'], inplace=True)
rankings.reset_index(inplace=True)
rankings.drop(columns=['index'], inplace=True)

rankings.rename(columns={'ranking': 'scopus_similarity_ranking'}, inplace=True)

rankings
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>year</th>
      <th>name</th>
      <th>faculty</th>
      <th>faculty_name</th>
      <th>query</th>
      <th>similarity_scopus_ranking</th>
      <th>embedding_duration</th>
      <th>vector_db_roundtrip</th>
      <th>charles_explorer_ranking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>491147</td>
      <td>2014.0</td>
      <td>Lexical semantic conversions in a valency lexicon</td>
      <td>11320</td>
      <td>Faculty of Mathematics and Physics</td>
      <td>lexical semantics</td>
      <td>0.0</td>
      <td>0.021093</td>
      <td>0.019749</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>592464</td>
      <td>2020.0</td>
      <td>Dirk Geeraerts:Theories of Lexical Semantics</td>
      <td>11210</td>
      <td>Faculty of Arts</td>
      <td>lexical semantics</td>
      <td>0.0</td>
      <td>0.021093</td>
      <td>0.019631</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>594640</td>
      <td>NaN</td>
      <td>DIACR-Ita @ EVALITA2020: Overview of the EVALI...</td>
      <td>-1</td>
      <td>Unknown faculty</td>
      <td>lexical semantics</td>
      <td>0.0</td>
      <td>0.021093</td>
      <td>0.015370</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>588808</td>
      <td>2020.0</td>
      <td>Dual semantics of intransitive verbs: lexical ...</td>
      <td>11210</td>
      <td>Faculty of Arts</td>
      <td>lexical semantics</td>
      <td>5.0</td>
      <td>0.021093</td>
      <td>0.015556</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>268937</td>
      <td>2000.0</td>
      <td>Manfred Stede: Lexical Semantics and Knowledge...</td>
      <td>11320</td>
      <td>Faculty of Mathematics and Physics</td>
      <td>lexical semantics</td>
      <td>2.0</td>
      <td>0.021093</td>
      <td>0.015275</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9264</th>
      <td>73261</td>
      <td>2005.0</td>
      <td>Modern X-ray imaging techniques and their use ...</td>
      <td>-1</td>
      <td>Unknown faculty</td>
      <td>biology</td>
      <td>94.0</td>
      <td>0.019262</td>
      <td>0.016730</td>
      <td>95</td>
    </tr>
    <tr>
      <th>9265</th>
      <td>31897</td>
      <td>2003.0</td>
      <td>Teaching tasks for biology education</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>18.0</td>
      <td>0.019262</td>
      <td>0.013611</td>
      <td>96</td>
    </tr>
    <tr>
      <th>9266</th>
      <td>566168</td>
      <td>2019.0</td>
      <td>Hands-on activities in biology: students' opinion</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>18.0</td>
      <td>0.019262</td>
      <td>0.012192</td>
      <td>97</td>
    </tr>
    <tr>
      <th>9267</th>
      <td>279719</td>
      <td>2013.0</td>
      <td>One of the Overlooked Themes in High School Pr...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>1.0</td>
      <td>0.019262</td>
      <td>0.013341</td>
      <td>98</td>
    </tr>
    <tr>
      <th>9268</th>
      <td>551485</td>
      <td>2018.0</td>
      <td>Evolutionary biology : In the history, today a...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>52.0</td>
      <td>0.019262</td>
      <td>0.013132</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
<p>9269 rows × 10 columns</p>
</div>



Using the original ranking positions and the predicted ranking positions as the source for the relevance feedback, we can calculate the nDCG (Normalized Discounted Cumulative Gain) score for the search result ranking.

The nDCG score is a measure of the ranking quality of the search results. It is calculated as the ratio of the DCG (Discounted Cumulative Gain) score to the IDCG (Ideal Discounted Cumulative Gain) score. The DCG score is calculated as the sum of the relevance scores of the search results, discounted by their position in the ranking. The IDCG score is the DCG score of the ideal ranking of the search results.

To transform the predicted Scopus rankings into relevance feedback, we introduce a new function $rel_q(d)$. For a given query $q$, the document $d$ is considered to have relevance of $rel_q(d)$, which is inverse proportional to its predicted ranking. This is necessary for the `nDCG` score calculation, which requires more relevant documents to have *higher* relevance scores.

The inverse proportionality is achieved by the following formula:

$$
rel_q(d) = \frac{a}{\text{rank}_q(d) + 1}
$$

where $a$ is a constant that scales the relevance scores and can help achieve better numerical stability of the `nDCG` score. For the purpose of this thesis, we set $a = 5$.
 The `+1` in the denominator is necessary to avoid division by zero, as our rankings are 0-based.

While it would be possible to achieve the ranking - relevance transformation via e.g. subtracting the predicted ranking from the total number of search results, our proposed method with $rel_q(d)$ introduces a non-linear transformation of the predicted rankings. This differentiates better between the search results that are ranked higher in the Scopus search results. This is in line with the human behaviour of having clearer opinions about the top search results. 

<!-- @ARTICLE{9357332,

  author={Su, Zhan and Lin, Zuyi and Ai, Jun and Li, Hui},

  journal={IEEE Access}, 

  title={Rating Prediction in Recommender Systems Based on User Behavior Probability and Complex Network Modeling}, 

  year={2021},

  volume={9},

  number={},

  pages={30739-30749},

  keywords={Motion pictures;Recommender systems;Predictive models;Prediction algorithms;Complex networks;Probability;Correlation;Complex network modeling;recommender systems;degree centrality;link prediction;user behavior probability},

  doi={10.1109/ACCESS.2021.3060016}} -->


```python
from typing import List
import numpy as np

def get_ndcg(gold_rankings: List[int]):
    dcg = 0
    for i, result in enumerate(gold_rankings):
        relevance = 5 / (result + 1)

        dcg += relevance / np.log2(i + 2)
    return dcg
```


```python
scores = pd.DataFrame()

queries = []
dcgs = []
best_ndcgs = []
for query in rankings['query'].unique():
    queries.append(query)
    dcgs.append(get_ndcg(rankings[rankings['query'] == query]['similarity_scopus_ranking'].to_list()))
    best_ndcgs.append(get_ndcg(rankings[rankings['query'] == query].sort_values('similarity_scopus_ranking')['similarity_scopus_ranking'].to_list()))

scores['queries'] = queries
scores['dcg'] = dcgs
scores['idcg'] = best_ndcgs
scores['ndcg'] = scores['dcg'] / scores['idcg']

scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>queries</th>
      <th>dcg</th>
      <th>idcg</th>
      <th>ndcg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lexical semantics</td>
      <td>29.482590</td>
      <td>33.308123</td>
      <td>0.885147</td>
    </tr>
    <tr>
      <th>1</th>
      <td>booster</td>
      <td>11.771938</td>
      <td>17.401799</td>
      <td>0.676478</td>
    </tr>
    <tr>
      <th>2</th>
      <td>logic</td>
      <td>4.557156</td>
      <td>11.233671</td>
      <td>0.405669</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gabapentin</td>
      <td>18.529698</td>
      <td>21.165926</td>
      <td>0.875449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>thalidomide</td>
      <td>13.151393</td>
      <td>20.222288</td>
      <td>0.650341</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>169</th>
      <td>sports medicine</td>
      <td>27.963706</td>
      <td>33.238549</td>
      <td>0.841303</td>
    </tr>
    <tr>
      <th>170</th>
      <td>dendrology</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>171</th>
      <td>alienism</td>
      <td>88.309044</td>
      <td>93.423651</td>
      <td>0.945254</td>
    </tr>
    <tr>
      <th>172</th>
      <td>hedonism</td>
      <td>42.712391</td>
      <td>47.876866</td>
      <td>0.892130</td>
    </tr>
    <tr>
      <th>173</th>
      <td>biology</td>
      <td>18.064385</td>
      <td>31.570575</td>
      <td>0.572191</td>
    </tr>
  </tbody>
</table>
<p>174 rows × 4 columns</p>
</div>




```python
scores.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dcg</th>
      <th>idcg</th>
      <th>ndcg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.919819</td>
      <td>19.167405</td>
      <td>0.761607</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.810894</td>
      <td>17.665142</td>
      <td>0.180979</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.094340</td>
      <td>0.094340</td>
      <td>0.405669</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.250473</td>
      <td>7.704989</td>
      <td>0.627563</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.527864</td>
      <td>14.840570</td>
      <td>0.736246</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.064385</td>
      <td>24.112511</td>
      <td>0.934206</td>
    </tr>
    <tr>
      <th>max</th>
      <td>104.693354</td>
      <td>104.693354</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



This gives us values for our benchmark - the nDCG values for the original Charles Explorer result ranking act as the baseline value.
We see that the mean nDCG score over all queries is 0.76, which is a good starting point for the benchmarking.

We can now proceed to collecting the graph data statistics for the publications in the search results. We will use the Charles Explorer API to get the graph data for the publications in the search results.

The following scripts collects betweenness centrality for 1-hop and 2-hop neighbourhoods of the publications in the search results. The "betweenness centrality" is a graph node measure that quantifies the importance of a node in a graph. It is calculated as the number of shortest paths between all pairs of nodes that pass through the node in question:

$$
g(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
$$

where $s$, $t$ and $v$ are nodes in the graph,
$\sigma_{st}$ is the number of shortest paths between nodes $s$ and $t$, and $\sigma_{st}(v)$ is the number of shortest paths between $s$ and $t$ that pass through $v$.


While this is often calculated for larger graphs, it is an useful measure for ego-networks too, as it can help us quantify the importance of a node in its local neighbourhood. Everett et. al have shown that the betweenness centrality might be strongly correlated with the actual global betweenness of a node in the graph.

<!-- @article{article,
author = {Everett, Martin and Borgatti, Stephen},
year = {2005},
month = {01},
pages = {31-38},
title = {Ego network betweenness},
volume = {27},
journal = {Social Networks},
doi = {10.1016/j.socnet.2004.11.007}
} -->


```python
from utils.memgraph import get_centralities
import pandas as pd

local_centrality = pd.DataFrame()

for i, query in enumerate(rankings['query'].unique()):
    if i % 10 == 0:
        print(f"{i}/{len(rankings['query'].unique())}")
    ids = rankings[rankings['query'] == query]['id'].astype(str).to_list()

    centralities = get_centralities(ids, 1, 2)
    centralities['query'] = query

    local_centrality = pd.concat([local_centrality, centralities])

local_centrality
```

    0/174
    10/174
    20/174
    30/174
    40/174
    50/174
    60/174
    70/174
    80/174
    90/174
    100/174
    110/174
    120/174
    130/174
    140/174
    150/174
    160/174
    170/174





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>centrality_1</th>
      <th>centrality_2</th>
      <th>query</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>74651</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>lexical semantics</td>
    </tr>
    <tr>
      <th>1</th>
      <td>97393</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>lexical semantics</td>
    </tr>
    <tr>
      <th>2</th>
      <td>117637</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>lexical semantics</td>
    </tr>
    <tr>
      <th>3</th>
      <td>118521</td>
      <td>0.002127</td>
      <td>0.000026</td>
      <td>lexical semantics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>145132</td>
      <td>0.000491</td>
      <td>0.002471</td>
      <td>lexical semantics</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>588322</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>96</th>
      <td>592720</td>
      <td>0.000313</td>
      <td>0.000168</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>97</th>
      <td>605136</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>98</th>
      <td>605641</td>
      <td>0.001118</td>
      <td>0.000168</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>99</th>
      <td>621001</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>biology</td>
    </tr>
  </tbody>
</table>
<p>9269 rows × 4 columns</p>
</div>



After calculating the ego-centralities, we merge the data with the original search results data. This gives us the final dataset for the benchmarking.


```python
local_centrality.to_csv('./local_centrality.csv', index=False)


centralities = rankings.join(local_centrality[['id', 'centrality_1', 'centrality_2']].set_index('id'), how='left', on='id', rsuffix='_centrality')
centralities
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>year</th>
      <th>name</th>
      <th>faculty</th>
      <th>faculty_name</th>
      <th>query</th>
      <th>similarity_scopus_ranking</th>
      <th>embedding_duration</th>
      <th>vector_db_roundtrip</th>
      <th>charles_explorer_ranking</th>
      <th>centrality_1</th>
      <th>centrality_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>491147</td>
      <td>2014.0</td>
      <td>Lexical semantic conversions in a valency lexicon</td>
      <td>11320</td>
      <td>Faculty of Mathematics and Physics</td>
      <td>lexical semantics</td>
      <td>0.0</td>
      <td>0.021093</td>
      <td>0.019749</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>592464</td>
      <td>2020.0</td>
      <td>Dirk Geeraerts:Theories of Lexical Semantics</td>
      <td>11210</td>
      <td>Faculty of Arts</td>
      <td>lexical semantics</td>
      <td>0.0</td>
      <td>0.021093</td>
      <td>0.019631</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>594640</td>
      <td>NaN</td>
      <td>DIACR-Ita @ EVALITA2020: Overview of the EVALI...</td>
      <td>-1</td>
      <td>Unknown faculty</td>
      <td>lexical semantics</td>
      <td>0.0</td>
      <td>0.021093</td>
      <td>0.015370</td>
      <td>2</td>
      <td>0.000545</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>588808</td>
      <td>2020.0</td>
      <td>Dual semantics of intransitive verbs: lexical ...</td>
      <td>11210</td>
      <td>Faculty of Arts</td>
      <td>lexical semantics</td>
      <td>5.0</td>
      <td>0.021093</td>
      <td>0.015556</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>268937</td>
      <td>2000.0</td>
      <td>Manfred Stede: Lexical Semantics and Knowledge...</td>
      <td>11320</td>
      <td>Faculty of Mathematics and Physics</td>
      <td>lexical semantics</td>
      <td>2.0</td>
      <td>0.021093</td>
      <td>0.015275</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9264</th>
      <td>73261</td>
      <td>2005.0</td>
      <td>Modern X-ray imaging techniques and their use ...</td>
      <td>-1</td>
      <td>Unknown faculty</td>
      <td>biology</td>
      <td>94.0</td>
      <td>0.019262</td>
      <td>0.016730</td>
      <td>95</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9265</th>
      <td>31897</td>
      <td>2003.0</td>
      <td>Teaching tasks for biology education</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>18.0</td>
      <td>0.019262</td>
      <td>0.013611</td>
      <td>96</td>
      <td>0.001878</td>
      <td>0.000670</td>
    </tr>
    <tr>
      <th>9266</th>
      <td>566168</td>
      <td>2019.0</td>
      <td>Hands-on activities in biology: students' opinion</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>18.0</td>
      <td>0.019262</td>
      <td>0.012192</td>
      <td>97</td>
      <td>0.001118</td>
      <td>0.000084</td>
    </tr>
    <tr>
      <th>9267</th>
      <td>279719</td>
      <td>2013.0</td>
      <td>One of the Overlooked Themes in High School Pr...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>1.0</td>
      <td>0.019262</td>
      <td>0.013341</td>
      <td>98</td>
      <td>0.006841</td>
      <td>0.057311</td>
    </tr>
    <tr>
      <th>9268</th>
      <td>551485</td>
      <td>2018.0</td>
      <td>Evolutionary biology : In the history, today a...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>52.0</td>
      <td>0.019262</td>
      <td>0.013132</td>
      <td>99</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>9609 rows × 12 columns</p>
</div>




```python
dcg_centrality_1 = []
dcg_centrality_2 = []

for query in scores['queries']:
    x = centralities[centralities['query'] == query].drop_duplicates(subset=['id'])
    dcg_centrality_1.append(
        get_ndcg(x.sort_values('centrality_1', ascending=False)['similarity_scopus_ranking'].to_list())
    )
    dcg_centrality_2.append(
        get_ndcg(x.sort_values('centrality_2', ascending=False)['similarity_scopus_ranking'].to_list())
    )
    

scores['dcg_1_centrality'] = dcg_centrality_1
scores['dcg_2_centrality'] = dcg_centrality_2
scores['ndcg_1_centrality'] = scores['dcg_1_centrality'] / scores['idcg']
scores['ndcg_2_centrality'] = scores['dcg_2_centrality'] / scores['idcg']
```


```python
scores.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dcg</th>
      <th>idcg</th>
      <th>ndcg</th>
      <th>dcg_1_centrality</th>
      <th>dcg_2_centrality</th>
      <th>ndcg_1_centrality</th>
      <th>ndcg_2_centrality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.919819</td>
      <td>19.167405</td>
      <td>0.761607</td>
      <td>13.883896</td>
      <td>13.944649</td>
      <td>0.698402</td>
      <td>0.705671</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.810894</td>
      <td>17.665142</td>
      <td>0.180979</td>
      <td>16.569196</td>
      <td>16.445045</td>
      <td>0.194843</td>
      <td>0.190947</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.094340</td>
      <td>0.094340</td>
      <td>0.405669</td>
      <td>0.094340</td>
      <td>0.094340</td>
      <td>0.390366</td>
      <td>0.390511</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.250473</td>
      <td>7.704989</td>
      <td>0.627563</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>0.539009</td>
      <td>0.545812</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.527864</td>
      <td>14.840570</td>
      <td>0.736246</td>
      <td>8.215773</td>
      <td>8.261666</td>
      <td>0.655002</td>
      <td>0.671576</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.064385</td>
      <td>24.112511</td>
      <td>0.934206</td>
      <td>16.277946</td>
      <td>15.838206</td>
      <td>0.873008</td>
      <td>0.884310</td>
    </tr>
    <tr>
      <th>max</th>
      <td>104.693354</td>
      <td>104.693354</td>
      <td>1.000000</td>
      <td>104.693354</td>
      <td>104.693354</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The mean nDCG values for the `1-` and `2-` hop betweenness centralities are noticeably lower than the baseline nDCG value for the basic Charles Explorer search result ranking. 

This is partially expected, since the relevance feedback is based on Scopus search result ranking, which is - presumably - based on the relevance of the search results to the query, i.e. the same metric we are using in the base Charles Explorer search result ranking.

In the next post, we will proceed with comparing the local graph statistics with the nDCG values based on the global popularity of the publications, as measured by the citation and reference count.
