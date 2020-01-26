# Automatic Continuation of User-Generated Item Lists 

### Introduction

This is the implementation of the paper as well as datasets and their splitting used in the paper:<br>
> Consistency-Aware Recommendation for User-Generated ItemList Continuation.<br>
> WSDM, 2020.<br>
> Yun He, Yin Zhang, Weiwen Liu and James Caverlee.<br>
> Department of Computer Science and Engineering, Texas A&M University.<br>
> Contact: yunhe@tamu.edu <br>
> Personal Page: http://people.tamu.edu/~yunhe/ <br>

User-generated item lists are popular on many platforms. Examples include song-based playlists on Spotify, image-based lists (or“boards”) on Pinterest, book-based lists on Goodreads, and answer-based lists on question-answer forums like Zhihu.
In these platforms, user-generated item lists are manually created, curated, and managed by users. 
Typically, users must first identify candidate items, determine if they are a good fit for a list, add them to a list, and then potentially provide ongoing updates
to the list (e.g., by adding or deleting items over time). To accelerate this process and assist users to explore more related items for their lists, 
we study the important yet challenging problem of user-generated item list continuation. That is, how can we recommend items that are related to the list and fit the user’s preferences?

### Usage 
We propose a novel model CAR to predict the next item that a user will possibly add into a list.

#### Input
The input of CAR includes: (1) the containing relationship between lists and items; (2) the creating relationship between users and lists.

#### (1) The containing relationship between lists and items.
``listId itemId``

which means that those items are curated in this list. These interactions are stored in \data folder, like AotM.txt.zip (unzip this file when you run our code). 

Note that the last item is regarded as test data, the item before the last item is regarded as validation data and the rest of items are treated as training data.

#### (2) The containing relationship between lists and items.
``key: listId, value: userId``

which means that the list is created by this user. These creating relationships are stored in \data folder, like AotM_creator_list.dict (python dictionary object).

#### Output
The output are the evaluation results comparing the ranked items for each list from CAR and the groundtruth (e.g., the last item of AotM.txt). For each test item, we sample 100 negative items. CAR predicts scores for the 100 negative items and 1 test item, and rank these 101 items based on their scores. The 100 negative items for each list are stored in \data folder, like AotMListItems_len5_item5_cut1000.negativeEvalTest.zip (unzip this file when you run our code).

#### Run
python main.py --dataset Zhihu --train_dir test

For more hyper-parameters, please see main.py.

### Citation
TBD
