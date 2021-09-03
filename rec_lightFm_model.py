import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightfm import LightFM
import os
import plotly.express as px
import umap.plot
import plotly.io as pio
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm.evaluation import auc_score
import seaborn as sns
import logging
import recfcn as rf
from lightfm.data import Dataset
logger = logging.getLogger("__main__")


class PreprocessingData:
    def __init__(self, item_features_file, item_user_iter_file):
        self._item_features_file = item_features_file
        self._item_user_iter_file = item_user_iter_file
        self.n_users = 0
        self.n_items = 0
        self.tar_cols =[]


    @staticmethod
    def lowercase_text(x):
        try:
            x = x.lower()
        except:
            pass
        return x

    @staticmethod
    def encoding(x):
        if x == 'Y':
            code = 1
        elif x == 'N':
            code = -1
        else:
            code = 0
        return code

    def load_data(self):
        # load datasets
        self._item_features = pd.read_excel(os.path.join('data', self._item_features_file))
        self._item_user_interactions = pd.read_excel(os.path.join('data', self._item_user_iter_file))

    def clean_data(self):
        # Cleaning item_features
        self._item_features.fillna('unk', inplace=True)
        item_features_cols = self._item_features.columns.tolist()
        remove_cols_list = ['Name ', 'Category', 'SubCategory', 'Cuisine']

        item_cat_cols = ['Category', 'SubCategory', 'Cuisine']
        # Categorical features
        for col in item_cat_cols:
            self._item_features[col] = self._item_features[col].apply(self.lowercase_text)

        item_features_clean = self._item_features.copy()
        for col in remove_cols_list: item_features_cols.remove(col)
        # Encoding categorical features
        for col in item_features_cols:
            item_features_clean[col] = item_features_clean[col].apply(self.encoding)

        # Numerical features
        numeric_item_features_clean = item_features_clean[item_features_cols]
        item_features_clean.drop('Cuisine', 1, inplace=True)
        item_features_clean.rename(columns = {'Name ':'business_name'}, inplace=True)

        # Cleaning item_user_interactions
        self._item_user_interactions.fillna(0, inplace=True)

        # Change format to review dataframe
        item_user_visits = self._item_user_interactions.copy()
        item_user_visits.set_index('Name ', inplace=True)
        item_user_visits = item_user_visits[item_user_visits == 1].stack().reset_index().drop(0, 1)
        item_user_visits.rename(columns={'level_1': 'business_name'}, inplace=True)

        item_user_visits_cols = item_features_cols

        add_cols_list = ['Name ', 'business_name']
        for col in add_cols_list: item_user_visits_cols.append(col)

        # create similar structure like Data review
        item_users_interactions = item_user_visits.merge(item_features_clean, on='business_name')

        item_users_interactions = item_users_interactions[item_user_visits_cols]
        item_users_interactions.rename(columns={'Name ': 'user_name'}, inplace=True)
        self.n_users = item_users_interactions['user_name'].unique().shape[0]
        self.n_items = item_users_interactions['business_name'].unique().shape[0]

        # Data User
        # check user interactions distribution
        tmp_user = item_users_interactions.groupby('user_name')['business_name'].count().reset_index()
        data_user = item_users_interactions.groupby('user_name').sum().reset_index()
        data_user = data_user.merge(tmp_user, on='user_name')
        data_user.rename(columns={'business_name': 'places_visited_count'}, inplace=True)
        data_user.to_csv('data/user_data_processed.csv')

        # Data Item
        tmp_item = item_users_interactions.groupby(['business_name']).count().reset_index()
        tmp_item.rename(columns={'user_name': 'visits_count'}, inplace=True)
        tmp_item = tmp_item[['business_name', 'visits_count']]
        item_features_clean = item_features_clean.merge(tmp_item, on='business_name', how='left')
        item_features_clean.fillna(0, inplace=True)
        item_users_interactions['implicit_feedback'] = 1
        item_users_interactions.to_csv('data/interactions.csv')

        return item_users_interactions, item_features_clean, data_user

    def dataset_model(self, data_user, item_users_interactions, item_features_clean):

        dataset = Dataset()
        dataset.fit(item_users_interactions.user_name, item_users_interactions.business_name)
        num_users, num_items = dataset.interactions_shape()
        print('Num users:{}, num_items{}.'.format(num_users, num_items))

        # Fit item and user features
        data_business = item_features_clean.copy()
        data_business.to_csv('data/item_features_processed.csv')

        dataset.fit_partial(items=data_business.business_name,
                            item_features=['visits_count'])

        # fit tar_cols
        tar_cols = [x for x in data_business.columns[3:-1]]
        self.tar_cols = tar_cols
        dataset.fit_partial(items=data_business.business_name,
                            item_features=tar_cols)
        # Users
        user_cols = [x for x in data_user.columns[1:]]
        dataset.fit_partial(users=data_user.user_name,
                            user_features=user_cols)

        # check data
        print(type(dataset))
        print(dataset.model_dimensions())
        print(dataset.user_features_shape())
        print(dataset.item_features_shape())
        print(dataset.interactions_shape())

        # look at item feature mapping
        a = dataset.mapping()[3]
        print(list(a.items())[0:10])

        interactions, weights = self.item_features_matrix(dataset, item_users_interactions)



        #build item features dataset
        max_b_rc = max(data_business.visits_count)
        item_features = dataset.build_item_features(((x['business_name'],
                                                      {'visits_count': 1 * x['visits_count'] / max_b_rc,
                                                       **self.build_dict(x, tar_cols, [1 * x['visits_count'] / max_b_rc])})
                                                     for index, x in data_business.iterrows()))
        # build user features dataset
        max_u_rc = max(data_user.places_visited_count)
        user_features = dataset.build_user_features(((x['user_name'],
                                                      {'places_visited_count': 1 * x['places_visited_count'] / max_u_rc,
                                                       **self.user_build_dict(x, user_cols,
                                                                         [1 * x['places_visited_count'] / max_u_rc])})
                                                     for index, x in data_user.iterrows()))

        print(repr(item_features))
        print(item_features.shape)

        print(repr(user_features))
        print(user_features.shape)

        # train-test split
        seed = 120
        train, test = random_train_test_split(interactions, test_percentage=0.2,
                                              random_state=np.random.RandomState(seed))

        print(f'The dataset has {train.shape[0]} users and {train.shape[1]} items, with {test.getnnz()} interactions in the test and {train.getnnz()} interactions in the training set.')

        print(train.multiply(test).nnz == 0)  # make sure train and test are truly disjoint

        return dataset, data_business, train, test, user_features, item_features

    @staticmethod
    def item_features_matrix(dataset, item_users_interactions):
        """
        Build a item features matrix out of an iterable of the form (item id, [list of feature names]) or
        (item id, {feature name: feature weight}).

        Parameters:
        data (iterable of the form) – (item id, [list of feature names]) or (item id, {feature name: feature weight}).
        Item and feature ids will be translated to internal indices constructed during the fit call.
        normalize (bool, optional) – If true, will ensure that feature weights sum to 1 in every row.
        Returns:
        feature matrix – Matrix of item features.

        Return type:
        CSR matrix (num items, num features)
        """

        # build interaction
        (interactions, weights) = dataset.build_interactions([(x['user_name'],
                                                               x['business_name'],
                                                               x['implicit_feedback']) for index, x in
                                                              item_users_interactions.iterrows()])

        print(repr(interactions))

        return interactions, weights

    @staticmethod
    # build user and item features
    def build_dict(df, tar_cols, val_list):
        rst = {}
        for col in tar_cols:
            rst[col] = df[col]
        sum_val = sum(list(rst.values()))  # get sum of all the tfidf values

        if (sum_val == 0):
            return rst
        else:

            w = (2 - sum(val_list)) / sum_val  # weight for each tag to be able to sum to 1
            for key, value in rst.items():
                rst[key] = value * w
        return rst

    @staticmethod
    def user_build_dict(df, tar_cols, val_list):
        rst = {}
        for col in tar_cols:
            rst[col] = df[col]
        sum_val = sum(list(rst.values()))  # get sum of all the tfidf values

        if (sum_val == 0):
            return rst
        else:
            w = (2 - sum(val_list)) / sum_val  # weight for each tag to be able to sum to 1
            for key, value in rst.items():
                rst[key] = value * w
        return rst

    @staticmethod
    def get_similar_places(model, tag_labels, places):
        # Define similarity as the cosine of the angle between the tag latent vectors
        l_features = list(set(tag_labels) - set(places))

        # Create a dataframe with columns as the name of places
        df = pd.DataFrame(columns=tag_labels, index=tag_labels)

        # Normalize the vectors to unit length
        tag_embeddings = (model.item_embeddings.T
                          / np.linalg.norm(model.item_embeddings, axis=1)).T

        for tag_id in range(0, len(tag_embeddings)):
            query_embedding = tag_embeddings[tag_id]
            similarity = np.dot(tag_embeddings, query_embedding)
            df.iloc[tag_id, :] = similarity
        df_sim_placces = df.drop(l_features, axis=1).drop(l_features, axis=0)
        return df, df_sim_placces

    @staticmethod
    def lightfm_model(train, test, loss_mod = 'logistic', num_threads=25, num_components=43,
                      num_epochs=18, item_alpha=2.88752e-6, lr=0.06652, k=4):
        seed = 120
        NUM_THREADS = num_threads
        NUM_COMPONENTS = num_components
        NUM_EPOCHS = num_epochs
        ITEM_ALPHA = item_alpha
        learning_rate = lr

        ##Pure Collaborative Filtering models

        model = LightFM(loss=loss_mod, random_state=seed,
                        item_alpha=ITEM_ALPHA,
                        no_components=NUM_COMPONENTS,
                        learning_rate=learning_rate)

        model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

        # Compute and print the AUC score
        train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
        print('Collaborative filtering train AUC: %s' % train_auc)

        test_auc = auc_score(model, test, num_threads=NUM_THREADS).mean()
        print('Collaborative filtering test AUC: %s' % test_auc)

        print("Train precision: %.4f" % precision_at_k(model, train, k=k, num_threads=NUM_THREADS).mean())
        print("Test precision: %.4f" % precision_at_k(model, test, train_interactions=train, k=k,
                                                      num_threads=NUM_THREADS).mean())
        return train_auc, test_auc, model

    @staticmethod
    def lightfm_model_hybrid(train, test, loss_mod='warp', num_threads=4, num_components=21,
                      num_epochs=16, item_alpha=5.97967e-6, lr=0.06652, k=4):
        seed = 120
        NUM_THREADS = num_threads
        NUM_COMPONENTS = num_components
        NUM_EPOCHS = num_epochs
        ITEM_ALPHA = item_alpha
        learning_rate = lr

        ##Pure Collaborative Filtering models

        model = LightFM(loss=loss_mod, random_state=seed,
                        item_alpha=ITEM_ALPHA,
                        no_components=NUM_COMPONENTS,
                        learning_rate=learning_rate)

        model = model.fit(train,user_features=user_features,
                          item_features=item_features,
                          epochs=NUM_EPOCHS,num_threads=NUM_THREADS)

        # Compute and print the AUC score
        train_auc = auc_score(model, train,user_features=user_features,
                              item_features=item_features, num_threads=NUM_THREADS).mean()

        print('Hybrid train AUC: %s' % train_auc)

        test_auc = auc_score(model, test,user_features=user_features,
                             item_features=item_features,num_threads=NUM_THREADS).mean()
        print('Hybrid test AUC:: %s' % test_auc)

        print("Train precision: %.4f" % precision_at_k(model, train,
                                               item_features=item_features,user_features=user_features, k=k,
                                               num_threads=NUM_THREADS).mean())
        print("Test precision: %.4f" % precision_at_k(model, test,train_interactions=train,
                                              item_features=item_features,user_features=user_features, k=k,
                                             num_threads=NUM_THREADS).mean())
        return train_auc, test_auc, model

    @staticmethod
    def similarity_user_items_ranks(model, train, test, data_meta, user_ids,
                                    name, mapping, tag=None, train_interactions=None,
                                    user_features=None, item_features=None,
                                    num_threads=2):
        """
        function to obtain the ranking venues
        """

        n_users, n_items = test.shape
        user_items_rank = pd.DataFrame(np.zeros([n_users, n_items]),
                                       columns=data_meta[name].unique())

        for user_name, user_id in user_ids.items():

            if train_interactions is None:
                scores = model.predict(user_id, np.arange(n_items),
                                       user_features=user_features,
                                       item_features=item_features,
                                       num_threads=num_threads)
                i_idx = [x for x in np.argsort(-scores)]
                top_items = data_meta.loc[i_idx, name]

                user_id_score = pd.DataFrame()
                user_id_score['Score'] = scores
                user_id_score['top venues'] = data_meta.loc[:, name]

                user_id_score.sort_values(by=['scores'], inplace=True, ascending=False)
                user_id_score.set_index('top venues', inplace=True)

            else:
                item_ids = np.delete(np.arange(n_items), train.tocsr()[user_id].indices)
                scores = model.predict(user_id, item_ids, user_features=user_features,
                                       item_features=item_features, num_threads=num_threads)
                i_idx = [x for x in np.argsort(-scores)]
                top_items = data_meta.loc[i_idx, name]

                user_id_score = pd.DataFrame()
                user_id_score['scores'] = scores
                user_id_score['top venues'] = data_meta.loc[:, name]

                user_id_score.sort_values(by=['scores'], inplace=True, ascending=False)
                user_id_score.set_index('top venues', inplace=True)

            print(" %s" % user_name)
            print(user_id_score.head(30))

            for c in top_items:
                user_items_rank.loc[user_id, c] = user_id_score.loc[c, 'scores']
                user_items_rank.loc[user_id, 'user'] = user_name

            print('-------------------------------------')
        user_items_rank.set_index('user', inplace=True)
        return user_items_rank

    def test_rec(self, model, dataset, train, test, user_features, item_features, data_business):
        rf.sample_train_recommendation(model, train, data_business, [1], 5, 'business_name',
                                       mapping=dataset.mapping()[2], tag='Category',
                                       user_features=user_features, item_features=item_features)
        user_index = list(set(rf.get_user_index(test)))

        rf.sample_test_recommendation(model, train, test, data_business, [5], 5, 'business_name',
                                      mapping=dataset.mapping()[2],
                                      train_interactions=train, tag='Category', user_features=user_features,
                                      item_features=item_features)

        tag_labels = list(dataset.mapping()[3].keys())

        #Example of labels
        target_ls = ['WiFi', 'DJs', 'Crafty Egg']
        for tag in target_ls:
            tag_id = tag_labels.index(tag)
            print(f'Most similar tags for {tag_labels[tag_id]}: {[tag_labels[x] for x in rf.get_similar_tags(model, tag_id)]}')
        # for each restaurant/bar , the complete similarity ranking of all other restaurants/bars
        tag_labels = list(dataset.mapping()[3].keys())
        places = list(dataset.mapping()[2].keys())
        df_similarities, df_similarities_places = self.get_similar_places(model, tag_labels, places)
        user_ids_all = dataset.mapping()[1]
        user_ids = {key:value for (key,value) in [x for x in user_ids_all.items()][0:self.n_users]}
        user_items_rank = self.similarity_user_items_ranks(model, train, test, data_business,
                                                      user_ids, 'business_name', mapping=dataset.mapping()[2],
                                                      train_interactions=train, tag='Category',
                                                      user_features=user_features,
                                                      item_features=item_features)
        print(user_items_rank)

        return user_items_rank, df_similarities, df_similarities_places

    # plot umap representations of a dataframe with columns
    @staticmethod
    def plot_2d_umap(df, features):

        mapper = umap.UMAP(metric='euclidean', n_components=2).fit(df[features])
        hover_data = pd.DataFrame({'index':df.index,
                                   'label':df['business_name'].fillna('None'),
                                   'item':df['business_name']})
        p = umap.plot.interactive(mapper, labels=df['Category'].fillna('None'), hover_data=hover_data, point_size=5)
        umap.plot.show(p)

    @staticmethod
    def plot_3d_umap(df, features):


        mapper = umap.UMAP(n_components=3).fit(df[features])
        df_3d = pd.DataFrame({'x':mapper.embedding_[:,0], 'y':mapper.embedding_[:,1], 'z':mapper.embedding_[:,2]})
        fig = px.scatter_3d(df_3d, x='x', y='y', z='z',
                            text=df['business_name'],
                            color=df['Category'].fillna('None'))
        pio.renderers.default = "browser"
        fig.show()
    def plot_rec(self, df_similarities_places, data_business, dataset, model):
        # plot item features using UMAP and dataframe data_business
        self.plot_2d_umap(data_business, self.tar_cols)

        # plot item embeddings using UMAP and dataframe data_business.
        self.plot_3d_umap(data_business, self.tar_cols)

        # Create visualization of user embeddings and item embeddings

        user_ids_all = dataset.mapping()[1]
        user_ids = {key: value for (key, value) in [x for x in user_ids_all.items()][0:self.n_users]}
        item_labels = list(dataset.mapping()[2].keys())
        user_embeddings = model.user_embeddings
        item_embeddings = model.item_embeddings
        item_labels_all = dataset.mapping()[3]

        # create dataframe for item embedding
        item_embeddings_df = pd.DataFrame(data=model.item_embeddings)
        item_embeddings_df['business_name'] = item_labels_all
        item_embeddings_df = item_embeddings_df[item_embeddings_df['business_name'].isin(item_labels)]
        item_embeddings_df = item_embeddings_df.merge(item_features_clean[['business_name', 'Category']],
                                                      on='business_name')
        tar_cols_items = [x for x in item_embeddings_df.columns[0:-2]]
        # Visualize item embeddings
        self.plot_2d_umap(item_embeddings_df, tar_cols_items)
        self.plot_3d_umap(item_embeddings_df, tar_cols_items)
        item_embeddings_df.plot(x=0, y=1, style='o')

        df_similarities_places = df_similarities_places.astype('float')
        plt.imshow(df_similarities_places.iloc[0:20, 0:20].values, cmap='RdYlBu')
        plt.colorbar()
        plt.xticks(range(len(df_similarities_places.iloc[0:20, 0:20])), df_similarities_places.iloc[0:20, 0:20].columns)
        plt.yticks(range(len(df_similarities_places.iloc[0:20, 0:20])), df_similarities_places.iloc[0:20, 0:20].index)
        plt.show()
        ax = sns.heatmap(df_similarities_places.iloc[0:20,0:20])


if __name__ == "__main__":
    logger.info('# -------------------------------------------------------------------------------------- #')
    logger.info('# --------------------------- Preprocessing Data --------------------------------------- #')
    logger.info('# -------------------------------------------------------------------------------------- #')
    item_features_file = 'item_feature_table.xlsx'
    item_user_iter_file = 'item_user_interactions.xlsx'
    prep = PreprocessingData(item_features_file,item_user_iter_file)
    # Load data
    prep.load_data()

    # Clean Data
    item_users_interactions, item_features_clean, data_user = prep.clean_data()

    logger.info('# -------------------------------------------------------------------------------------- #')
    logger.info('# --------------------------- Build Lightfm Data --------------------------------------- #')
    logger.info('# -------------------------------------------------------------------------------------- #')

    # Build Lightfm Dataset
    dataset, data_business, train, test, user_features, item_features = prep.dataset_model(data_user, item_users_interactions, item_features_clean)

    # Build Lightfm Model

    logger.info('# -------------------------------------------------------------------------------------- #')
    logger.info('# --------------------------- Build Lightfm Model --------------------------------------- #')
    logger.info('# -------------------------------------------------------------------------------------- #')

    # Logistic
    train_auc, test_auc, model_log = prep.lightfm_model(train, test, loss_mod='logistic')

    # brp
    train_auc, test_auc, model_brp = prep.lightfm_model(train, test, loss_mod='bpr', num_threads=25, num_components=45,
                                            num_epochs=43, item_alpha=1.3846e-6, lr=0.016144)
    # warp
    train_auc, test_auc, model_warp = prep.lightfm_model(train, test, loss_mod='warp', num_threads=25, num_components=21,
                                             num_epochs=16, item_alpha=5.97967e-6, lr=0.033)
    # Warp hybrid model
    train_auc, test_auc, model_hybrid_warp = prep.lightfm_model_hybrid(train,test, loss_mod='warp')

    logger.info('# -------------------------------------------------------------------------------------- #')
    logger.info('# --------------------------- Test Recommendation -------------------------------------- #')
    logger.info('# -------------------------------------------------------------------------------------- #')

    user_items_rank, df_similarities, df_similarities_places = prep.test_rec(model_hybrid_warp, dataset, train, test, user_features, item_features, data_business)
    print(user_items_rank)

    logger.info('# -------------------------------------------------------------------------------------- #')
    logger.info('# ------------------------ Plot uma Representations ------------------------------------ #')
    logger.info('# -------------------------------------------------------------------------------------- #')
    prep.plot_rec(df_similarities_places, data_business, dataset, model_hybrid_warp)




