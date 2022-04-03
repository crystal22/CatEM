from cfg.option import Options
from dataset import SequenceContextDataset
from model import Trainer
from utils import unserialize

# 读取配置文件
config_file = './cfg/config1.cfg'
config = Options(config_file)


def get_poi_info(poi_info_path):
    f = open(poi_info_path, 'r', encoding='utf-8')

    content = f.readline()
    category2count = dict()
    while content != '':
        items = content.strip().split(',')
        category2count[items[1]] = category2count.get(items[1], 0) + 1
        content = f.readline()
    f.close()

    leq_5_category = []
    for category in category2count.keys():
        if category2count[category] <= 5:
            leq_5_category.append(category)

    return leq_5_category, category2count


def get_sequence_info(sequence_info_path):
    """
    输入为某个数据集POI信息的路径
    :return: {venue_id: [lon, lat, category]}
    """
    # sequence_info_path = 'E:\\Documents\\DataSet\\UbiComp2016\\' + config.city.upper() \
    #                      + 'Data\\Checkins-Sequence-' + config.city.upper() + '.csv'

    category_set = set()
    sequence_info = []
    num_checkins = 0
    # leq_5_category = get_poi_info()
    # f = open(sequence_info_path, 'r', encoding='utf-8')
    f1 = open(sequence_info_path, 'r', encoding='utf-8')
    content = f1.readline()
    while content != '':
        items = content.strip().split(',')
        sequence_info.append(items)
        temp = [items[0]]
        num_checkins += len(items[1:])
        for item in items[1:]:
            category = item[item.index('#') + 1: item.index('@')]
            # if category in leq_5_category:
            #     continue
            temp.append(item)
            category_set.add(category)
        # f1.write(','.join(temp) + '\n')
        content = f1.readline()
    f1.close()
    print('check-ins:' + str(num_checkins))
    category_order = list(category_set)
    category2id = dict(zip(sorted(category_order), range(len(category_order))))

    return sequence_info, category2id


def main():
    # get_poi_info()
    config = unserialize('./configs/config1.json')
    sequence_info, category2id = get_sequence_info(config["sequence_info_path"])
    dataset = SequenceContextDataset(config, sequence_info, category2id, config["rk_path"])
    Trainer(dataset, config)


if __name__ == '__main__':
    main()
