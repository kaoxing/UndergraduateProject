# author: kaoing
# 处理nnUNet训练结果summary.json，统计6类非边缘结果的dice
import json


def handle_summary(path_json):
    dice = 0.0
    num = 0
    with open(path_json, 'r') as file:
        data = json.load(file)
        metric_per_case: list = data['metric_per_case']
        for case in metric_per_case:
            metrics: dict = case['metrics']
            for key in metrics.keys():
                if int(key) > 6:
                    continue
                dice += metrics[key]['Dice']
                num += 1
        print("average_dice", dice/num)
        

if __name__ == '__main__':
    handle_summary('summary_edge_1.json')
