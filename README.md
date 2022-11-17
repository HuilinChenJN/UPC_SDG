
# Privacy-Preserving Synthetic Data Generation for Recommendation Systems

This is our implementation for the paper:

Fan Liu, Zhiyong Cheng, Huilin Chen, Yinwei Wei, Liqiang Nie, and Mohan Kankanhalli. 2022. [Privacy-Preserving Synthetic Data Generation for Recommendation Systems](https://doi.org/10.1145/3477495.3532044). In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 1379–1389. 




Please cite our SIGIR'22 paper if you use our codes. Thanks!
## Updates
- Update (November 17, 2022) This update shares our training dataset with public researchers. Please check the Dataset section.

- Update (June 11, 2022)
This update will integrate the function of manually generating user privacy settings（`process_data.py`） into the model. Now you can specify the user's privacy sensitivity by setting the parameter `--privacy_ratio` or `--privacy_settings_json`

### Table of contents
1. [Requirement](#enviroment-requirement)
2. [Dataset](#Dataset)
3. Usage
   - [Run UPC-SDG Model](#Run-UPC-SDG-Model)
   - [Evaluate model effectiveness](#Evaluate-model-effectiveness)
4. [Genereated train data](#Genereated-train-data)
5. [Results](#Results)

## Enviroment Requirement

1. Install via pip: `pip install -r requirements.txt`
2. Create the empty folders, `output` and `data`.
3. Download the train data from the [Amazon Review Data](http://jmcauley.ucsd.edu/data/amazon/links.html) 
and [SNAP](https://snap.stanford.edu/data/loc-gowalla.html) Page, details setting see [Dataset](#dataset) section 
4. Prepare for pre-trained User/Item embedding weight from [Google Ddrive](https://drive.google.com/drive/folders/14bI4GXyK2VZIROn3BGSHljrFWdqud3WU?usp=sharing) and put them in `./code/embedding`

## Dataset

We provide three processed datasets: Office, Clothing and Gowalla. Besides, we also share our training dataset [Google Ddrive](https://drive.google.com/drive/folders/1FuiaWIVQhbsMwIUErmmDzfSh_ib1iH73?usp=sharing) with public researchers.

|#Interactions|#Users|#Items|#interactions|sparsity|
|:-|:-|:-|:-|:-|
|Office|4,874|2,405|52,957| 99.55%|
|Clothing|18,209| 17,317| 150,889| 99.95%|
|Gowalla| 29,858| 40,981| 1,027,370| 99.91%|



-`train.txt` Train file. Each line is a user with her/his positive interactions with items: (userID and itemID)
-`test.txt`Test file. Each line is a user with her/his several positive interactions with items: (userID and itemID)
-`user_privacy.json` User's privacy setting. Each element is user's sensitivity of privacy guarantee for original items.


**Note:**
IF you need to add other dataset, please consider below steps:
1. Add additional dataset data into `data` folder, includes `train.txt` and `test.txt`
2. Add new dataset name in `./code/world.py` and `./code/register.py`
3. Extend `dataloader.py` file If you need

## Run UPC-SDG Model

Run UPC-SDG model to generate new train data considering user privacy sensitivity, 
and different dataset parameters are shown below:


Run model on **Office** dataset:

```bash 
 python -u ./code/main.py --decay=1e-1 --lr=0.001 --seed=2022 --dataset="Office" --topks="[20]" --recdim=64 --bpr_batch=2048 --load=1 --replace_ratio=0.2 --privacy_ratio=0.1 --bpr_loss_d=1 --similarity_loss_d=3
```

run model on **Clothing** dataset:

```bash 
python -u ./code/main.py --decay=1e-1 --lr=0.001 --seed=2022 --dataset="Clothing" --topks="[20]" --recdim=64 --bpr_batch=2048 --load=1 --replace_ratio=0.2 --privacy_ratio=0.1 --bpr_loss_d=1 --similarity_loss_d=3
```

run model on **Gowalla** dataset:

```bash 
python -u ./code/main.py --decay=1e-3 --lr=0.001 --seed=2022 --dataset="gowalla" --topks="[20]" --recdim=64 --bpr_batch=2048 --load=1 --replace_ratio=0.2 --privacy_ratio=0.1 --bpr_loss_d=1 --similarity_loss_d=3
```


Extend:Set user privacy settings(Optional)

If you need to load the special user's privacy settings, you can set path parameter  into run command. （e.g. `--privacy_settings_json='./data/privacy_example.json'`）

Besides, we provide `process_data.py` to generate `user_privacy.json` file into dataset folder:

```bash 
python process_data.py --data_path="Office" --privacy_ration=0.7
```

*Note*:
- `data_path`: the path for the train data folder, which include `train.txt/text.txt` files.
- `privacy_ration`: is defined as privacy sensitivity for the original item, limit is (0,1), (e.g. for validating, we used 0.1, 0.3, 0.5, 0.7, 0.9 respectively in our paper).



## Evaluate model effectiveness

When the training process of UPC-SDG model is finished, the model will output the new train data into `output` folder, name format is `{dataset name}-replace{replace ratio}-{output prefix}.txt`, then the new train file and original test file as privacy guarantee dataset input into other recommendation system 
to evaluate (e.g. BPRMF, NeuMF and LightGCN ).

*You can find the generated train data used to evaluate in the `output` folder, or generate a new file according to needed*

## Genereated train data

For convenient, we provided the genereated train data used in our paper. you can get them from [Google Ddrive](https://drive.google.com/drive/folders/1Z6-Ux4Cot_LLCeHuG2Blme7y-59O4P-m?usp=sharing) and put them into other recommendation system to evaluate.


## Results
*All metrics is under top-20*

![results](https://s1.ax1x.com/2022/07/16/j4T6aQ.png)
