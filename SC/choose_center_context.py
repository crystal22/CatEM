# !/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from geopy.distance import geodesic
import pandas as pd
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')


class KNearestNeighbors:

    def __init__(self, k, poi_info, output_path):
        """
        :param k:
        :param poi_info: dict() {poi_id:[Longitude, latitude, venue_category_name], ...}
        """
        self.k = k
        self.poi_info = poi_info
        self.output_path = output_path
        self.grid2poi = dict()
        self.poi2grid = dict()
        self.get_k_nearest_neighbors()

    def get_neighbors(self, center_poi, neighbor_width):
        context_pois = []
        center_grid = self.poi2grid[center_poi]
        for x in range(center_grid[0] - neighbor_width, center_grid[0] + neighbor_width + 1):
            for y in range(center_grid[1] - neighbor_width, center_grid[1] + neighbor_width + 1):
                if (x, y) in self.grid2poi.keys():
                    context_pois.extend(self.grid2poi[(x, y)])
        context_pois.remove(center_poi)
        return context_pois

    def get_k_nearest_neighbors(self):
        max_latitude = max_longitude = -math.inf
        min_latitude = min_longitude = math.inf
        for poi in self.poi_info.keys():
            latitude, longitude, category = self.poi_info[poi]
            if max_latitude <= latitude and max_longitude <= longitude:
                max_latitude = latitude
                max_longitude = longitude

            if min_latitude >= latitude and min_longitude >= longitude:
                min_latitude = latitude
                min_longitude = longitude

        for poi in tqdm(self.poi_info.keys(), desc='griding'):
            latitude, longitude, _ = self.poi_info[poi]
            x = int((latitude - min_latitude) * 100 // 1)
            y = int((longitude - min_longitude) * 100 // 1)
            self.poi2grid[poi] = (x, y)
            if (x, y) not in self.grid2poi.keys():
                self.grid2poi[(x, y)] = [poi]
            else:
                self.grid2poi[(x, y)].append(poi)

        for center_poi in tqdm(self.poi_info.keys(), desc='writing contexts'):
            latitude1, longitude1, _ = self.poi_info[center_poi]

            neighbor_width = 0
            candidate_list = []
            while len(candidate_list) < self.k:
                candidate_list = self.get_neighbors(center_poi, neighbor_width)
                neighbor_width += 1

            contexts = dict()
            for poi in candidate_list:
                latitude2, longitude2, _ = self.poi_info[poi]
                dis = geodesic((latitude1, longitude1), (latitude2, longitude2)).m
                contexts[poi] = dis
            contexts = sorted(contexts.items(), key=lambda x: x[1], reverse=False)[:self.k]
            with open(self.output_path, 'a') as f:
                f.write(center_poi + ",")
                for context in contexts[:self.k - 1]:
                    f.write(str(context[0]) + ",")
                f.write(str(contexts[self.k - 1][0]))
                f.write('\n')
