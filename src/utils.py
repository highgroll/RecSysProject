import pandas as pd
import numpy as np
import os, sys
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from src.metrics import precision_at_k
from src.utils import prefilter_items

def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    top_cheapest_item = data['price'].quantile(0.20).item_id.tolist()
    data = data[~data['item_id'].isin(top_cheapest_item)]

    # Уберем слишком дорогие товары
    top_price_item = data['price'].quantile(0.99995).item_id.tolist()
    data = data[~data['item_id'].isin(top_price_item)]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    user_item_matrix = pd.pivot_table(data,
                                      index='user_id', columns='item_id',
                                      values='quantity',
                                      aggfunc='count',
                                      fill_value=0
                                      )

    user_item_matrix = user_item_matrix.astype(float)

    sparse_user_item = csr_matrix(user_item_matrix).tocsr()

    return data


def postfilter_items(user_id, recommednations):
    pass