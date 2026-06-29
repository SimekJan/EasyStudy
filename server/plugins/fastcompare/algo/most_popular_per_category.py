import numpy as np
from collections import defaultdict
import random

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    PreferenceElicitationBase,
    DataLoaderBase,
    Parameter,
    ParameterType
)


class MostPopularPerCategory(AlgorithmBase):
    def __init__(self, loader, positive_threshold=2.5, **kwargs):

        print("_____INITIALIZING_MODEL_____", flush=True)

        self._categories = loader.get_all_categories()
        self._item_index_categories = loader.get_item_index_categories
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._rating_matrix = (
            self._loader.ratings_df
            .pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._popularity = np.sum(self._rating_matrix > 2.5, axis=0)
        self._category_to_items = defaultdict(list)

        for i in range(len(self._popularity)):
            categories = self._item_index_categories(i)
            for c in categories:
                self._category_to_items[c].append(i)

    def fit(self):

        print("_____TRAINING_____", flush=True)

        pass

    def get_best_item_by_category(self, category):
        if category not in self._category_to_items:
            return None

        item_indices = self._category_to_items[category]
        best_idx = max(item_indices, key=lambda i: self._popularity[i])

        return self._all_items[best_idx], self._popularity[best_idx]

    def predict(self, selected_items, filter_out_items, k):

        print("_____PREDICTING_____", flush=True)
        print("Selected items: ", flush=True)
        for item in selected_items:
            print(item, flush=True)

        selected_set = set(selected_items)
        filter_set = set(filter_out_items)

        category_scores = {
            cat: sum(self._popularity[i] for i in items)
            for cat, items in self._category_to_items.items()
        }

        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        categories = [c for c, _ in sorted_categories]

        if k <= len(categories):
            chosen_categories = categories[:k]
        else:
            chosen_categories = categories[:]
            weights = [category_scores[c] for c in categories]

            chosen_categories.extend(
                random.choices(categories, weights=weights, k=k - len(categories))
            )

        used_items = set()
        used_categories = defaultdict(int)

        result = []

        for cat in chosen_categories:
            candidates = self._category_to_items.get(cat, [])

            sorted_items = sorted(
                candidates,
                key=lambda i: (
                    i in selected_set,      # avoid previously selected
                    i in used_items,        # avoid duplicates in output
                    -self._popularity[i]    # prefer popular
                )
            )

            chosen_item = None

            for i in sorted_items:
                if i in filter_set:
                    continue
                chosen_item = i
                break

            if chosen_item is None:
                continue

            print("For category:", cat, "was chosen:", self._all_items[chosen_item])

            result.append(self._all_items[chosen_item])
            used_items.add(chosen_item)
            used_categories[cat] += 1

            if len(result) == k:
                break

        return result

    @classmethod
    def name(cls):
        return "Most popular per category"

    @classmethod
    def parameters(cls):
        return []