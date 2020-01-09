# mlb-ie
This repo contains the code for the evaluation scripts for [mlb dataset](https://github.com/ratishsp/mlb-data-scripts)

## Creating IE json
The json file containing the ground-truth information to be used for IE is in a slightly different format from [mlb dataset](https://github.com/ratishsp/mlb-data-scripts). Hence the json needs to be generated using the script [ie_json_creation.py](https://github.com/ratishsp/mlb-ie/blob/master/ie_json_creation.py). 
```
python ie_json_creation.py -input_folder "../data/mlb/json" -output_folder "../data/mlb/ie_json" -type [train|valid|test] -scoring
```
The argument scoring indicates the json will contain scoring play-by-plays. During the training of IE models,
the scoring argument is turned off. As the models presented in [paper](https://www.aclweb.org/anthology/P19-1195/) make use of scoring play-by-play, when evaluating model summaries the scoring version of IE json is used.

## Information/Relation Extraction

### Creating Training/Validation Data
You can create a dataset for training or evaluating the relation extraction system as follows:

```
python mlb_data_utils.py -mode make_ie_data -input_path "../data/mlb/ie_json" -output_fi "mlb-ie.h5"
```

This will create files `mlb-ie.h5`, `mlb-ie.dict`, and `mlb-ie.labels`.

### Evaluating Generated summaries
1. You can download the extraction models we ensemble to do the evaluation from this [link](https://drive.google.com/open?id=1n0GIfeo6iGQGUEbB-ZQjBKIEbhQ3kHVC). There are two models in total, with the name pattern `*ie-ep*`. Put these extraction models in the same directory as `extractor.lua`. (Note that `extractor.lua` hard-codes the paths to these saved models, so you'll need to change this if you want to substitute in new models.)

2. Once you've generated summaries, you can put them into a format the extraction system can consume as follows:

```
python mlb_data_utils.py -mode prep_gen_data -gen_fi mlb_ent-beam5_gens.txt -dict_pfx "mlb-ie" -output_fi mlb_ent-beam5_gens.h5 -input_path "../data/mlb/ie_json"
```

where the file you've generated is called `mlb_ent-beam5_gens.txt` and the dictionary and labels files are in `mlb-ie.dict` and `mlb-ie.labels` respectively (as above). This will create a file called `mlb_ent-beam5_gens.h5`, which can be consumed by the extraction system.

3. The extraction system can then be run as follows:

```
th extractor.lua -gpuid 1 -datafile mlb-ie.h5 -preddata mlb_ent-beam5_gens.h5 -dict_pfx "mlb-ie" -just_eval
```

This will print out the **RG** metric numbers. (For the recall number, divide the 'nodup correct' number by the total number of generated summaries, e.g., 1739). It will also generate a file called `mlb_ent-beam5_gens.h5-tuples.txt`, which contains the extracted relations, which can be compared to the gold extracted relations.

4. We now need the tuples from the gold summaries. `val_mlb.h5-tuples.txt` and `test_mlb.h5-tuples.txt` have been included in the repo, but they can be recreated by repeating steps 2 and 3 using the gold summaries (with one gold summary per-line, as usual).

5. The remaining metrics can now be obtained by running:

```
python non_rg_metrics.py val_mlb.h5-tuples.txt mlb_ent-beam5_gens.h5-tuples.txt
```

# Acknowledgements
Most of the code in this repo and the README have been adapted from harvardnlp data2text [repo](https://github.com/harvardnlp/data2text) with changes done to handle mlb dataset.

