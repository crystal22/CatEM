import time


def generate_dataset(poi_info_path, sequence_info_path, generate_dataset_path):
    poi_info = {}
    f = open(poi_info_path, 'r', encoding='utf-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')
        poi_info[items[0]] = [float(items[2]), float(items[3]), items[1]]
        content = f.readline()
    f.close()

    f = open(sequence_info_path, 'r', encoding='utf-8')

    f1 = open(generate_dataset_path, 'w', encoding='utf-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')
        # sequence_info.append(items)
        user_id = items[0]
        for item in items[1:]:
            venue = item[: item.index('#')]
            category = item[item.index('#') + 1: item.index('@')]
            time_stamp = item[item.index('@') + 1:]
            time_struct = time.gmtime(int(time_stamp) / 1000)
            date = str(time_struct.tm_year) + '-' + str(time_struct.tm_mon) + '-' + str(time_struct.tm_mday) \
                   + ' ' + str(time_struct.tm_hour) + ':' + str(time_struct.tm_min) + ':' + str(time_struct.tm_sec)
            # print(user_id + ',' + category + ','
            #          + str(poi_info[venue][0]) + ',' + str(poi_info[venue][1]) + ',' + date + '\n')
            f1.write(user_id + ',' + category + ','
                     + str(poi_info[venue][0]) + ',' + str(poi_info[venue][1]) + ',' + date + '\n')

        content = f.readline()
    f.close()
    f1.close()


def main():
    city = 'nyc'
    """
        poi_info: id, category, lat, lon
        for example: 4d124470f898b1f71491d781,Coffee Shop,40.615782,-74.016313
    """
    poi_info_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + city.upper() + 'Data\\Pois-' + city.upper() + '.csv'
    """
        poi_info: user id, check-in, check-in
        check-in for example: 4ace6c89f964a52078d020e3#Airport@1354644894000
    """
    sequence_info_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + city.upper() \
                         + 'Data\\Checkins-Sequence-' + city.upper() + '.csv'
    generate_dataset_path = './data/dataset_UbiComp2016_' + city.upper() + '_for_GeoSAN.txt'
    """
        dataset_UbiComp2016_NYC_for_GeoSAN.txt
        date format: <user_id, category, lat, lng, %Y-%m-%d %H:%M:%S>
    """
    generate_dataset(poi_info_path, sequence_info_path, generate_dataset_path)


if __name__ == '__main__':
    main()

