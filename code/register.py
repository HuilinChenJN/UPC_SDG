import world
import dataloader
import model
import utils
from pprint import pprint



if world.config['privacy_settings_json'] is not None:
    # specify the json file of privacy settings
    world.config['privacy_ratio'] = 'externalJsonFile'
elif world.config['privacy_ratio'] is not None:
    # set the valid privacy settings
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