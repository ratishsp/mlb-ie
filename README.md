# mlb-ie
This repo contains the code for the evaluation scripts for [mlb dataset](https://github.com/ratishsp/mlb-data-scripts)

## Creating IE json
The json file containing the ground-truth information to be used for IE is in a slightly different format from [mlb dataset](https://github.com/ratishsp/mlb-data-scripts). Hence the json needs to be generated using the script [ie_json_creation.py](https://github.com/ratishsp/mlb-ie/blob/master/ie_json_creation.py). 
```
IE_ROOT=<Root Path for IE>
python ie_json_creation.py -input_folder "../data/mlb/json" -output_folder "${IE_ROOT}/ie_json" -type [train|valid|test]
```

Alternatively the json files can be downloaded from https://drive.google.com/drive/folders/1q9xpjIBkF7YOerXE6eSiSDq158kjw8Nn?usp=sharing
## Information/Relation Extraction

### Creating Training/Validation Data
You can create a dataset for training or evaluating the relation extraction system as follows:

```
The files to map ordinal adjective to innings can be downloaded from https://drive.google.com/drive/folders/19u99YQqoG7K4xIq9XdeZgPGr2jvQSCIe?usp=sharing
ORDINAL_ADJECTIVE_MAP_FOLDER=<Path of downloaded ordinal adjective map files>

python mlb_data_utils.py -mode make_ie_data -input_path ${IE_ROOT}/ie_json -output_fi "mlb-ie.h5" -ordinal_inning_map_file ${ORDINAL_ADJECTIVE_MAP_FOLDER}
```

This will create files `mlb-ie.h5`, `mlb-ie.dict`, and `mlb-ie.labels`.

### Evaluating Generated summaries
1. You can download the extraction models we ensemble to do the evaluation from this [link](https://drive.google.com/drive/folders/1q9xpjIBkF7YOerXE6eSiSDq158kjw8Nn?usp=sharing). There are two models in total, with the name pattern `*ie-ep*`. Put these extraction models in the same directory as `extractor.lua`. (Note that `extractor.lua` hard-codes the paths to these saved models, so you'll need to change this if you want to substitute in new models.)

2. Once you've generated summaries, you can put them into a format the extraction system can consume as follows:

```
DOC_GEN=<path_to_doc_gen_repo>
GEN=$DOC_GEN/gen
IDENTIFIER=mlb
python mlb_data_utils.py -mode prep_gen_data -gen_fi $DOC_GEN/gen/$IDENTIFIER-segment-beam5_gens.txt \
    -dict_pfx "$IE_ROOT/data/mlb-ie" -output_fi $DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5 \
    -input_path $IE_ROOT/ie_json -ordinal_inning_map_file $DOC_GEN/gen/$IDENTIFIER-inning-map-beam5_gens.txt
```

where the file you've generated is called `mlb-beam5_gens.txt` and the dictionary and labels files are in `mlb-ie.dict` and `mlb-ie.labels` respectively (as above). This will create a file called `mlb-beam5_gens.h5`, which can be consumed by the extraction system.

3. The extraction system can then be run as follows:

```
th extractor.lua -gpuid 1 -datafile $IE_ROOT/data/mlb-ie.h5 \
    -preddata $DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5 -dict_pfx "$IE_ROOT/data/mlb-ie" -just_eval \
    -ignore_idx 14
```

This will print out the **RG** metric numbers. (For the recall number, divide the 'nodup correct' number by the total number of generated summaries, e.g., 1739). It will also generate a file called `mlb-beam5_gens.h5-tuples.txt`, which contains the extracted relations, which can be compared to the gold extracted relations.

4. We now need the tuples from the gold summaries. `val_mlb.h5-tuples.txt` and `test_mlb.h5-tuples.txt` have been included in the repo, but they can be recreated by repeating steps 2 and 3 using the gold summaries (with one gold summary per-line, as usual).

5. The remaining metrics can now be obtained by running:

```
python non_rg_metrics.py val_mlb.h5-tuples.txt mlb-beam5_gens.h5-tuples.txt
```
Note this command requires Python 3.

# Acknowledgements
Most of the code in this repo and the README have been adapted from harvardnlp data2text [repo](https://github.com/harvardnlp/data2text) with changes done to handle mlb dataset.

