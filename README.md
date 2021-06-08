# lightfm-recsys
An example recsys task to complete.

### The task: 

Implement a simple recommender system using the (LightFM)[https://making.lyst.com/lightfm/docs/home.html] package with a custom dataset. The model will aim to recommend restaurants and bars (venues) to users using an app. Some sample data has been collected for the task. The sample data will consist of an item-feature table, describing the attributes of each venue, as well as a user-item table, depicting interactions between users and venues. 

We're looking for data transformation to the format expected by the model, followed by a training step, evaluation and predictions. The result should be a **hybrid*** model that is evaluated and is able to run predictions for new users. 

You may wish to do some EDA to understand the dataset, but it is not required. 

Code quality will be assessed. We're looking for evidence that you can write production code, but perfection is not required. 

Please feel free to try different things or experiment with this. Have fun, see what you can do.

To complete the task, make a Pull Request to this repository. 

*A hybrid recommendation system is one that uses both collaborative filtering and content filtering. It is worth getting familiar with these terms.

You may find the documentation and these guides helpful: 

- LightFM documentation: https://making.lyst.com/lightfm/docs/home.html  

https://towardsdatascience.com/how-i-would-explain-building-lightfm-hybrid-recommenders-to-a-5-year-old-b6ee18571309
https://github.com/lyst/lightfm/blob/master/examples/stackexchange/hybrid_crossvalidated.ipynb
https://towardsdatascience.com/creating-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-cc8b431618af
https://towardsdatascience.com/solving-business-usecases-by-recommender-system-using-lightfm-4ba7b3ac8e62

These helper functions may also be of use: 
https://github.com/aayushmnit/cookbook/blob/master/recsys.py
