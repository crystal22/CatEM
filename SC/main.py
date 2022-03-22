from choose_center_context import KNearestNeighbors
from dataset import SpatialContextDataset
from model import SCTrainer
from cfg.option import Options

# 读取配置文件
config_file = './cfg/example.cfg'
params = Options(config_file)


def get_poi_info():
    """
    输入为某个数据集POI信息的路径
    :return: {venue_id: [lon, lat, category]}
    """
    poi_info_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + params.city.upper() \
                    + 'Data\\Pois_revise-' + params.city.upper() + '.csv'

    poi_info = {}
    f = open(poi_info_path, 'r', encoding='utf-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')
        poi_info[items[0]] = [float(items[2]), float(items[3]), items[1]]
        content = f.readline()
    f.close()
    return poi_info


def main():
    # 保存每个poi最近的k个邻居的路径
    context_neighbors_path = '.\\knn_neighbors\\context_neighbors' + str(params.k) + '_' + params.city + '.txt'

    # 需要将数据集中文件转换为这种形式 {venue_id: [lon, lat, category]}
    poi_info = get_poi_info()
    # 得到每个POI的k邻近之后可以注释掉下面代码 对于每个k只需要执行一次
    # KNearestNeighbors(params.k, poi_info, context_neighbors_path)
    knn_dataset = SpatialContextDataset(poi_info, context_neighbors_path)
    SCTrainer(knn_dataset, params)


if __name__ == '__main__':
    main()
