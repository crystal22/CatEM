from tqdm import tqdm
from utils import serialize, unserialize
import json


def get_category_to_type(hierarchy_tree_path):
    category_to_type = {}
    # Arts & Entertainment/College & University/Food/
    # Nightlife Spot/Outdoors & Recreation/Professional & Other Places/
    # Residence/Shop & Service/Travel & Transport
    category_to_id = {'4d4b7104d754a06370d81259': 0, '4d4b7105d754a06372d81259': 1, '4d4b7105d754a06374d81259': 2,
                      '4d4b7105d754a06376d81259': 3, '4d4b7105d754a06377d81259': 4, '4d4b7105d754a06375d81259': 5,
                      '4e67e38e036454776db1fb3a': 6, '4d4b7105d754a06378d81259': 7, '4d4b7105d754a06379d81259': 8,
                      '4d4b7105d754a06373d81259': 9}

    with open(hierarchy_tree_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in tqdm(lines):

        temp = line.split(',')
        category_name, category_id, category_type = temp[1], temp[0], temp[2]

        if '$' not in category_type:
            category_to_type[category_name] = category_to_id[category_id]
        else:
            temp = category_type.split('$')
            category_to_type[category_name] = category_to_id[temp[len(temp) - 2]]
    return category_to_type


if __name__ == '__main__':
    hierarchy_tree_path = 'E:\\JupyterNoteBookSpace\\Hierarchical Tree\\data\\category.csv'
    category_to_type = get_category_to_type(hierarchy_tree_path)
    file_name = './data/CategoryToType.json'
    with open(file_name, 'w') as f:
        json.dump(category_to_type, f)
