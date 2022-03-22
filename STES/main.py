from dataset import SequenceContextDataset
from model import STESTrainer
from cfg.option import Options

# 读取配置文件
config_file = './cfg/example.cfg'
params = Options(config_file)


def get_sequence_info():
    """
    输入为某个数据集POI信息的路径
    :return: {venue_id: [lon, lat, category]}
    """
    # sequence_info_path = 'E:\\Documents\\DataSet\\UbiComp2016\\' + params.city.upper() \
    #                      + 'Data\\Checkins-Sequence-' + params.city.upper() + '.csv'

    category_set = set()
    sequence_info = []
    num_checkins = 0
    # leq_5_category = get_poi_info()
    # f = open(sequence_info_path, 'r', encoding='utf-8')
    f1 = open('E:\\Bins\\DataSet\\UbiComp2016\\' + params.city.upper() \
              + 'Data\\Checkins-Sequence-filter-leq5-' + params.city.upper() + '.csv', 'r', encoding='utf-8')
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
    # 需要将数据集中文件转换为这种形式 {venue_id: [lon, lat, category]}
    sequence_info, category2id = get_sequence_info()
    dataset = SequenceContextDataset(sequence_info, params, category2id)
    STESTrainer(dataset, params)


if __name__ == '__main__':
    main()
