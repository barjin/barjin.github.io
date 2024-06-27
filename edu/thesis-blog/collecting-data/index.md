<( [Selecting benchmark queries](../ranking-benchmarks/index.md) )

# Collecting the gold-standard data for benchmarking

> This Python notebook shows the process of benchmarking the search result ranking for the Charles Explorer application.
> It is a part of my diploma thesis at the Faculty of Mathematics and Physics, Charles University, Prague.
>
> Find out more about the thesis in the [GitHub repository](https://github.com/barjin/master-thesis).
>
> Made by Jindřich Bär, 2024. 

In the previous post, we have picked the best queries for our benchmarking. To recap, we'll be comparing the search results ranking between (different configurations of) Charles Explorer and the Elsevier Scopus search engine. 

Our target is to explore the usability of "local" (university-wide) graph data about publications, their authors and their collaborations for improving the search result ranking in the Charles Explorer application.
We're using Elsevier Scopus as the benchmark, as it is a widely used and well-established search engine for scientific publications - and contains data about citations and other metrics that we don't have in our local graph data.

For this purpose, we have selected a set of queries returning a representative set of results (their distribution among the faculties is close to the distribution of the whole dataset). We will run these queries in both search engines and compare the results.

To start, we can load both the dataset of the queries and the dataset of the search results from the default Charles Explorer configuration.


```python
import pandas as pd

queries = pd.read_csv('./best_queries.csv');
charles_explorer_results = pd.read_csv('./search_results.csv');

# Search results contain a superset of the queries in the best queries file.
# We begin by filtering the search results to only include the queries in the best queries file.
charles_explorer_results[charles_explorer_results['query'].isin(queries['0'])].to_csv('./filtered_search_results.csv', index=False);
```


```python
charles_explorer_results = pd.read_csv('./filtered_search_results.csv')

charles_explorer_results[charles_explorer_results['query'] == 'thrombolytic']['name']
```




    7603     Thrombolytic therapy in acute pulmonary embolism
    7604    Guidelines for Intravenous Thrombolytic Therap...
    7605    Pulmonary embolism - thrombolytic and anticoag...
    7606    Pulmonary embolism - thrombolytic and anticoag...
    7607    Thrombolytic therapy of acute central retinal ...
                                  ...                        
    7685    Intravenous Thrombolysis in Unknown-Onset Stro...
    7686    Variable penetration of primary angioplasty in...
    7687    Prospective Single-Arm Trial of Endovascular M...
    7688    "Stent 4 Life" Targeting PCI at all who will b...
    7689    How does the primary coronary angioplasty effe...
    Name: name, Length: 87, dtype: object



An experiment showed that for this set of publications for the query `thrombolytic`, only 3 of the 30 above were listed in the top 50 most relevant search results in Elsevier Scopus. 

To provide more complete data for the benchmark, we sample the top 100 search results for each query in both search engines - so we get better insight into whether some publications are missing or whether they are just ranked lower.

Since we've first only collected the top 30 search results, we now have to rerun the result collection process again (on the seleted queries).


```python
from utils.charles_explorer import get_dataframe_for_queries

charles_explorer_results = get_dataframe_for_queries(queries['0'].to_list())
charles_explorer_results.to_csv('./filtered_search_results.csv', index=False)
```

    Processed 0 queries
    Processed 30 queries
    Processed 60 queries
    Processed 90 queries
    Processed 120 queries
    Processed 150 queries


The dataset is now missing the ranking order for the search results. The records are however sorted by the "relevance" score (the search result ordering is the same as in the application), so we can easily add the ranking order back.


```python
charles_explorer_results = pd.read_csv('./filtered_search_results.csv')

charles_explorer_results['ranking'] = None

for query in charles_explorer_results['query'].unique():
    charles_explorer_results.loc[charles_explorer_results['query'] == query, 'charles_explorer_position'] = range(1, len(charles_explorer_results.loc[charles_explorer_results['query'] == query]) + 1)
```


```python
charles_explorer_results[charles_explorer_results['query'] == 'biology']
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
      <th>ranking</th>
      <th>charles_explorer_position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2404</th>
      <td>478570</td>
      <td>2005.0</td>
      <td>Developmental biology for medics</td>
      <td>11130</td>
      <td>Second Faculty of Medicine</td>
      <td>biology</td>
      <td>None</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>129678</td>
      <td>2010.0</td>
      <td>Basic statistics for biologists (Statistics wi...</td>
      <td>11320</td>
      <td>Faculty of Mathematics and Physics</td>
      <td>biology</td>
      <td>None</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2406</th>
      <td>439661</td>
      <td>2001.0</td>
      <td>Symposium 'Electromagnetic Aspects of Selforga...</td>
      <td>11510</td>
      <td>Faculty of Physical Education and Sport</td>
      <td>biology</td>
      <td>None</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2407</th>
      <td>121104</td>
      <td>2009.0</td>
      <td>Jesuit´s other face: Bohuslaus Balbinus as bio...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>None</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2408</th>
      <td>37559</td>
      <td>1998.0</td>
      <td>Review of - Carbohydrates - Structure and Biology</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>None</td>
      <td>5.0</td>
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
    </tr>
    <tr>
      <th>2499</th>
      <td>73261</td>
      <td>2005.0</td>
      <td>Modern X-ray imaging techniques and their use ...</td>
      <td>-1</td>
      <td>Unknown faculty</td>
      <td>biology</td>
      <td>None</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>2500</th>
      <td>31897</td>
      <td>2003.0</td>
      <td>Teaching tasks for biology education</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>None</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>2501</th>
      <td>566168</td>
      <td>2019.0</td>
      <td>Hands-on activities in biology: students' opinion</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>None</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>2502</th>
      <td>279719</td>
      <td>2013.0</td>
      <td>One of the Overlooked Themes in High School Pr...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>None</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2503</th>
      <td>551485</td>
      <td>2018.0</td>
      <td>Evolutionary biology : In the history, today a...</td>
      <td>11310</td>
      <td>Faculty of Science</td>
      <td>biology</td>
      <td>None</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>



## Loading the Elsevier Scopus search results

We start by loading the dataset of the search results from the Elsevier Scopus search engine.

The Scopus *Advanced search* feature allows us to use a special query language to submit the search queries. This query language offers a set of Prolog-like functors\[[1](https://www.dai.ed.ac.uk/groups/ssp/bookpages/quickprolog/node9.html#SECTION00044000000000000000)\], each connected to a specific attribute - or a set of attributes - of the publication record. The attributes of these functors are used in a substring search on the specified fields.

Apart from this, the query language also supports logical operators, such as `AND`, `OR`, and `AND NOT`.


We will use two of the available functors: `TITLE-ABS-KEY` and `AF-ID`:
- `TITLE-ABS-KEY` searches for the specified substring in the title, abstract, and keywords of the publication record. In this regard, it is similar to the full-text search in Charles Explorer, which searches in the same fields.
- `AF-ID` filters the search results by the affiliation ID of the author. This is useful for filtering the search results to only those publications where at least one of the authors is affiliated with Charles University.
Since Elsevier Scopus contains many records not affiliated with Charles University (but Charles Explorer only contains such records), this will help us to get a more comparable sets of search results.

By calling the Scopus API, we can get the search results in JSON format. We can then parse the JSON and load the search results into a pandas `DataFrame`.


```python
import json
import subprocess
import pandas as pd

def get_query_string(query):
    return f"AF-ID ( 60016605 ) AND TITLE-ABS-KEY ( \"{query}\" )"

def get_query_object(query, limit=10, offset=0):
    return {
        "documentClassificationEnum": "primary",
        "query": get_query_string(query),
        "sort": "r-f",
        "itemcount": limit,
        "offset": offset,
        "showAbstract": False
    }

def get_curl_call(query, limit=10, offset=0):
    return f"""curl\
 'https://www.scopus.com/api/documents/search'\
 --compressed\
 -X POST\
 -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0'\
 -H 'Accept: */*'\
 -H 'Accept-Language: en,cs;q=0.7,en-US;q=0.3'\
 -H 'Accept-Encoding: gzip, deflate, br, zstd'\
 -H 'content-type: application/json'\
 -H 'Origin: https://www.scopus.com'\
 -H 'Connection: keep-alive'\
 -H 'Cookie: ######################################### Cookies have been removed for security reasons. #########################################' \
 -H 'Sec-Fetch-Dest: empty'\
 -H 'Sec-Fetch-Mode: cors'\
 -H 'Sec-Fetch-Site: same-origin'\
 -H 'Priority: u=4'\
 -H 'Pragma: no-cache'\
 -H 'Cache-Control: no-cache'\
 -H 'TE: trailers' \
 --data-raw '{json.dumps(get_query_object(query, limit, offset))}'"""

def call_scopus_api(query, limit=10, offset=0):
    result = subprocess.run(get_curl_call(query, limit, offset), check=True, shell=True, stdout=subprocess.PIPE)
    return json.loads(result.stdout.decode('utf-8'))

df = pd.DataFrame()

for query in queries['0']:
    print(query)
    response = call_scopus_api(query, limit=100)
    rankings = pd.DataFrame.from_dict(response['items'])

    rankings['query'] = query
    rankings['ranking'] = range(0, len(rankings))

    df = pd.concat([
        df, 
        rankings    
    ])
    df.to_csv('./scopus_results.csv', index=False)
```


```python
import ast

df = pd.read_csv('./scopus_results.csv')
df['citationCount'] = df['citations'].apply(lambda x: ast.literal_eval(x).get('count'))
df['referenceCount'] = df['references'].apply(lambda x: ast.literal_eval(x).get('count'))

df
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
      <th>ranking</th>
      <th>links</th>
      <th>citations</th>
      <th>references</th>
      <th>totalAuthors</th>
      <th>freetoread</th>
      <th>eid</th>
      <th>subjectAreas</th>
      <th>authors</th>
      <th>...</th>
      <th>abstractAvailable</th>
      <th>publicationStage</th>
      <th>sourceRelationship</th>
      <th>pubYear</th>
      <th>databaseDocumentIds</th>
      <th>titles</th>
      <th>source</th>
      <th>title</th>
      <th>citationCount</th>
      <th>referenceCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>physics</td>
      <td>0</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 0, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 11, 'link': 'https://www.scopus.com/...</td>
      <td>2.0</td>
      <td>False</td>
      <td>2-s2.0-85130750924</td>
      <td>[{'code': 31, 'displayName': 'Physics and Astr...</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '', 'volume': '2458', 'articleNumber...</td>
      <td>2022.0</td>
      <td>{'SCP': '85130750924', 'PUI': '638086293', 'SC...</td>
      <td>['Ideas of Various Groups of Experts as a Star...</td>
      <td>{'active': True, 'publisher': 'American Instit...</td>
      <td>Ideas of Various Groups of Experts as a Starti...</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>physics</td>
      <td>1</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 1, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 3, 'link': 'https://www.scopus.com/a...</td>
      <td>2.0</td>
      <td>True</td>
      <td>2-s2.0-85072187757</td>
      <td>[{'code': 31, 'displayName': 'Physics and Astr...</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '1', 'volume': '1286', 'articleNumbe...</td>
      <td>2019.0</td>
      <td>{'SCP': '85072187757', 'PUI': '629310532', 'SC...</td>
      <td>['Practical Course on School Experiments for F...</td>
      <td>{'active': True, 'publisher': 'Institute of Ph...</td>
      <td>Practical Course on School Experiments for Fut...</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>physics</td>
      <td>2</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 0, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 2, 'link': 'https://www.scopus.com/a...</td>
      <td>4.0</td>
      <td>False</td>
      <td>2-s2.0-85099716344</td>
      <td>[{'code': 31, 'displayName': 'Physics and Astr...</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>False</td>
      <td>final</td>
      <td>{'issue': '', 'volume': '', 'articleNumber': '...</td>
      <td>2020.0</td>
      <td>{'SCP': '85099716344', 'PUI': '633981292', 'SC...</td>
      <td>['Collection of solved physics problems and co...</td>
      <td>{'active': False, 'publisher': 'Slovak Physica...</td>
      <td>Collection of solved physics problems and coll...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>physics</td>
      <td>3</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 8, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 77, 'link': 'https://www.scopus.com/...</td>
      <td>2.0</td>
      <td>False</td>
      <td>2-s2.0-85099862464</td>
      <td>[{'code': 33, 'displayName': 'Social Sciences'}]</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '4', 'volume': '43', 'articleNumber'...</td>
      <td>2021.0</td>
      <td>{'SNGEO': '2021003302', 'SCP': '85099862464', ...</td>
      <td>['Physics demonstrations: who are the students...</td>
      <td>{'active': True, 'publisher': 'Routledge', 'pu...</td>
      <td>Physics demonstrations: who are the students a...</td>
      <td>8</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>physics</td>
      <td>4</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 0, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 17, 'link': 'https://www.scopus.com/...</td>
      <td>4.0</td>
      <td>False</td>
      <td>2-s2.0-85130747100</td>
      <td>[{'code': 31, 'displayName': 'Physics and Astr...</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '', 'volume': '2458', 'articleNumber...</td>
      <td>2022.0</td>
      <td>{'SCP': '85130747100', 'PUI': '638086272', 'SC...</td>
      <td>['Use of the Collection of Solved Problems in ...</td>
      <td>{'active': True, 'publisher': 'American Instit...</td>
      <td>Use of the Collection of Solved Problems in Ph...</td>
      <td>0</td>
      <td>17</td>
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
      <th>7810</th>
      <td>specific</td>
      <td>95</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 3, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 96, 'link': 'https://www.scopus.com/...</td>
      <td>1.0</td>
      <td>True</td>
      <td>2-s2.0-85053545422</td>
      <td>[{'code': 33, 'displayName': 'Social Sciences'}]</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '3', 'volume': '50', 'articleNumber'...</td>
      <td>2018.0</td>
      <td>{'SCP': '85053545422', 'PUI': '623972267', 'SC...</td>
      <td>['Academia without contention? The legacy of C...</td>
      <td>{'active': True, 'publisher': 'Sociologicky Us...</td>
      <td>Academia without contention? The legacy of Cze...</td>
      <td>3</td>
      <td>96</td>
    </tr>
    <tr>
      <th>7811</th>
      <td>specific</td>
      <td>96</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 8, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 24, 'link': 'https://www.scopus.com/...</td>
      <td>2.0</td>
      <td>False</td>
      <td>2-s2.0-34547786691</td>
      <td>[{'code': 23, 'displayName': 'Environmental Sc...</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '2', 'volume': '55', 'articleNumber'...</td>
      <td>2007.0</td>
      <td>{'SCP': '34547786691', 'PUI': '47225913', 'SNC...</td>
      <td>['Specific pollution of surface water and sedi...</td>
      <td>{'active': True, 'publisher': '', 'publication...</td>
      <td>Specific pollution of surface water and sedime...</td>
      <td>8</td>
      <td>24</td>
    </tr>
    <tr>
      <th>7812</th>
      <td>specific</td>
      <td>97</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 0, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 56, 'link': 'https://www.scopus.com/...</td>
      <td>2.0</td>
      <td>False</td>
      <td>2-s2.0-85193778000</td>
      <td>[{'code': 33, 'displayName': 'Social Sciences'...</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '', 'volume': '', 'articleNumber': '...</td>
      <td>2023.0</td>
      <td>{'SCP': '85193778000', 'SRC-OCC-ID': '95648141...</td>
      <td>['Reflective approaches to professionalisation...</td>
      <td>{'active': False, 'publisher': 'Springer Inter...</td>
      <td>Reflective approaches to professionalisation t...</td>
      <td>0</td>
      <td>56</td>
    </tr>
    <tr>
      <th>7813</th>
      <td>specific</td>
      <td>98</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 0, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 0, 'link': 'https://www.scopus.com/a...</td>
      <td>1.0</td>
      <td>False</td>
      <td>2-s2.0-85185678534</td>
      <td>[{'code': 33, 'displayName': 'Social Sciences'}]</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '', 'volume': '14', 'articleNumber':...</td>
      <td>2023.0</td>
      <td>{'SCP': '85185678534', 'SRC-OCC-ID': '95444925...</td>
      <td>['THE CONCEPT OF DUE DILIGENCE IN THE CONTEXT ...</td>
      <td>{'active': True, 'publisher': 'Czech Society o...</td>
      <td>THE CONCEPT OF DUE DILIGENCE IN THE CONTEXT OF...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7814</th>
      <td>specific</td>
      <td>99</td>
      <td>[{'rel': 'self', 'type': 'GET', 'href': 'https...</td>
      <td>{'count': 1, 'link': 'https://www.scopus.com/a...</td>
      <td>{'count': 70, 'link': 'https://www.scopus.com/...</td>
      <td>3.0</td>
      <td>False</td>
      <td>2-s2.0-84937715409</td>
      <td>[{'code': 16, 'displayName': 'Chemistry'}]</td>
      <td>[{'links': [{'rel': 'self', 'type': 'GET', 'hr...</td>
      <td>...</td>
      <td>True</td>
      <td>final</td>
      <td>{'issue': '7', 'volume': '109', 'articleNumber...</td>
      <td>2015.0</td>
      <td>{'SCP': '84937715409', 'SNCHEM': '2015122725',...</td>
      <td>['Protease-activated receptors: Activation, in...</td>
      <td>{'active': True, 'publisher': 'Czech Society o...</td>
      <td>Protease-activated receptors: Activation, inhi...</td>
      <td>1</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
<p>7815 rows × 24 columns</p>
</div>




```python
df.to_csv("./scopus_results.csv", index=False)
```

With the data loaded into the `DataFrame`, we can now explore e.g. the correlation between the numerical attributes of the search results and the relevance score.


```python
df.select_dtypes(('int', 'float')).corr()
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
      <th>ranking</th>
      <th>totalAuthors</th>
      <th>scopusId</th>
      <th>pubYear</th>
      <th>citationCount</th>
      <th>referenceCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ranking</th>
      <td>1.000000</td>
      <td>0.038005</td>
      <td>0.081229</td>
      <td>0.109848</td>
      <td>0.062467</td>
      <td>0.053487</td>
    </tr>
    <tr>
      <th>totalAuthors</th>
      <td>0.038005</td>
      <td>1.000000</td>
      <td>0.033948</td>
      <td>0.040538</td>
      <td>0.113336</td>
      <td>0.094358</td>
    </tr>
    <tr>
      <th>scopusId</th>
      <td>0.081229</td>
      <td>0.033948</td>
      <td>1.000000</td>
      <td>0.806411</td>
      <td>0.015393</td>
      <td>0.243830</td>
    </tr>
    <tr>
      <th>pubYear</th>
      <td>0.109848</td>
      <td>0.040538</td>
      <td>0.806411</td>
      <td>1.000000</td>
      <td>0.033019</td>
      <td>0.283521</td>
    </tr>
    <tr>
      <th>citationCount</th>
      <td>0.062467</td>
      <td>0.113336</td>
      <td>0.015393</td>
      <td>0.033019</td>
      <td>1.000000</td>
      <td>0.218415</td>
    </tr>
    <tr>
      <th>referenceCount</th>
      <td>0.053487</td>
      <td>0.094358</td>
      <td>0.243830</td>
      <td>0.283521</td>
      <td>0.218415</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the `ranking` column is only very weakly correlated with the `citationCount` and `referenceCount` columns. Moreover, the `ranking` column is mostly correlated with the `pubYear` column (correlation coefficient `0.11`). This suggests that the default Scopus ranking is mostly influenced by the full-text search and does not take much reranking into account.
