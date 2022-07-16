import world
import dataloader
import model
import utils
from pprint import pprint



if world.config['privacy_settings_json'] is not None:
    # 指定了特定的json文件
    world.config['privacy_ratio'] = 'externalJsonFile'
elif world.config['privacy_ratio'] is not None:
    # 设置了有效的敏感度
    jsonPath = utils.generate_user_privacy_settings(world.dataset, world.config['privacy_ratio'])
    world.config['privacy_settings_json'] = jsonPath
else:
    raise ValueError(f"Haven't valid privacy settings")
#
# print( world.config['privacy_ratio'],  world.config['privacy_settings_json'])
#
# exit()



if world.dataset in ['Office', 'gowalla', 'yelp2018', 'KS10', 'Clothing', 'CD-Vinyl', 'amazon-book']:
    dataset = dataloader.Loader(path="./data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

# MODELS = {
#     'mf': model.PureMF,
#     # 'lgn': model.LightGCN
# }