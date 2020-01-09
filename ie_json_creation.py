# -*- coding: utf-8 -*-
import os
import json
import codecs
import argparse
N_A = "N/A"

batting_attrib = ["first_name", "last_name", "a","ab","avg","bb","cs","e","h","hbp","hr",
                  "obp","po","pos","r","rbi","sb","sf","slg","so"]
pitching_attrib = ["bb","er","era","h","hr","l","loss","s", "np","r",
                   "save","so","sv","w","win", "ip1", "ip2"]
pitching_attrib = ["p_"+entry for entry in pitching_attrib]
total_attrib = batting_attrib
total_attrib.extend(pitching_attrib)
ls_keys = ["team_hits", "team_errors", "result", "team_name", "team_city"]

NUM_PLAYERS  = 46
HOME = "HOME"
AWAY = "AWAY"
DELIM = u"￨"


def create_play_by_play_entry_json(key, new_key, inning_play, line_upd, greater_than_zero=False):
    if key in inning_play and ((greater_than_zero and int(inning_play[key]) > 0) or not greater_than_zero):
        if str(inning_play[key]) not in line_upd[new_key]:
            line_upd[new_key].append(str(inning_play[key]))


def create_play_by_play_entry_player_json(key, play_upd, entity, full_name_map):
    play_upd[key.replace(" ", "_")][full_name_map[entity]] = "Y"


def create_play_by_play_entry_player_append_value_json(key, inning_play, play_upd, entity, full_name_map):
    if full_name_map[entity] not in play_upd[key]:
        play_upd[key][full_name_map[entity]] = []
    if key in inning_play and int(inning_play[key]) > 0:
        if str(inning_play[key]) not in play_upd[key][full_name_map[entity]]:
            play_upd[key][full_name_map[entity]].append(str(inning_play[key]))


def get_play_by_play(entry, full_name_map, scoring):
    plays = entry["play_by_play"]
    vis_line_upd = {}
    vis_line_upd["team_runs"] = []
    vis_line_upd["runs"] = []
    vis_line_upd["error_runs"] = []
    #TODO innings?
    home_line_upd = {}
    home_line_upd["team_runs"] = []
    home_line_upd["runs"] = []
    home_line_upd["error_runs"] = []

    play_upd = {}
    play_upd["b1"] = {}
    play_upd["b2"] = {}
    play_upd["b3"] = {}
    play_upd["fielder_error"] = {}
    play_upd["fielder_error_scored"] = {}
    play_upd["single"] = {}
    play_upd["double"] = {}
    play_upd["triple"] = {}
    play_upd["home_run"] = {}
    play_upd["walk"] = {}
    play_upd["single_pitcher"] = {}
    play_upd["double_pitcher"] = {}
    play_upd["triple_pitcher"] = {}
    play_upd["home_run_pitcher"] = {}
    play_upd["walk_pitcher"] = {}
    play_upd["scorer"] = {}
    play_upd["rbi"] = {}

    for inning in range(1, len(entry['home_line']['innings'])+1):
        for top_bottom in ["top", "bottom"]:
            inning_plays = plays[str(inning)][top_bottom]
            for inning_play in inning_plays:
                if scoring and inning_play["runs"] == 0:  # exclude the play if non scoring
                    continue
                create_play_by_play_entry_json("home_team_runs", "team_runs", inning_play, home_line_upd)
                create_play_by_play_entry_json("away_team_runs", "team_runs", inning_play, vis_line_upd)

                for baserunner_key in ["b1", "b2", "b3"]:
                    if baserunner_key in inning_play and len(inning_play[baserunner_key])>0 and inning_play[baserunner_key][0] != N_A:
                        for baserunner_instance in inning_play[baserunner_key]:
                            create_play_by_play_entry_player_json(baserunner_key, play_upd, baserunner_instance,
                                                                  full_name_map)
                if 'event2' in inning_play and inning_play['event2'] == 'Error' and 'fielder_error' in inning_play:
                    create_play_by_play_entry_player_json("fielder_error", play_upd, inning_play["fielder_error"],
                                                          full_name_map)
                elif inning_play["event"]=='Field Error' and 'fielder_error' in inning_play :
                    create_play_by_play_entry_player_json("fielder_error", play_upd, inning_play["fielder_error"],
                                                          full_name_map)
                elif 'fielder_error' in inning_play :
                    create_play_by_play_entry_player_json("fielder_error", play_upd, inning_play["fielder_error"],
                                                          full_name_map)
                else:
                    if inning_play["event"].lower().replace(" ","_") in ["single", "double", "triple", "home_run", "walk"]:
                        create_play_by_play_entry_player_json(inning_play["event"].lower(), play_upd,
                                                              inning_play["batter"], full_name_map)
                        create_play_by_play_entry_player_json(inning_play["event"].lower() + "_pitcher", play_upd,
                                                              inning_play["pitcher"],
                                                              full_name_map)
                if "scorers" in inning_play and len(inning_play["scorers"])>0:
                    for scorer in inning_play["scorers"]:
                        create_play_by_play_entry_player_json("scorer", play_upd, scorer, full_name_map)
                if "batter" in inning_play:
                    create_play_by_play_entry_player_append_value_json("rbi", inning_play, play_upd,
                                                                       inning_play["batter"], full_name_map)
                create_play_by_play_entry_json("runs", "runs", inning_play,
                                               vis_line_upd if top_bottom == "top" else home_line_upd,
                                               greater_than_zero=True)
                create_play_by_play_entry_json("error_runs", "error_runs", inning_play,
                                               vis_line_upd if top_bottom == "top" else home_line_upd,
                                               greater_than_zero=True)
                if "error_runs" in inning_play and "fielder_error" in inning_play:
                    create_play_by_play_entry_player_json("fielder_error_scored", play_upd,
                                                          inning_play["fielder_error"], full_name_map)
    return home_line_upd, vis_line_upd, play_upd


def process(input_folder, output_folder, type, scoring):
    for filename in os.listdir(input_folder):
        if not type in filename:
            continue
        d = None
        with open(os.path.join(input_folder, filename)) as json_data:
            d = json.load(json_data)
        json_data.close()
        print 'filename', filename
        output_json_filename = os.path.join(output_folder, filename)
        output_json = []
        for entry in d:
            output_json_entry = entry
            full_name = output_json_entry["box_score"]["full_name"]
            full_name_map = {}
            for full_name_key, full_name_value in full_name.iteritems():
                if full_name_value != N_A:
                    full_name_map[full_name_value] = full_name_key

            home_line_upd, vis_line_upd, play_upd = get_play_by_play(entry, full_name_map, scoring)
            for key in ["home_line", "vis_line"]:
                output_json_entry[key]["team_city"] = entry[key]["team_city"]
            output_json_entry["vis_city"] = entry["vis_city"]
            output_json_entry["home_city"] = entry["home_city"]
            output_json_entry["home_line_upd"] = home_line_upd
            output_json_entry["vis_line_upd"] = vis_line_upd
            output_json_entry["play_upd"] = play_upd
            output_json.append(output_json_entry)

        with codecs.open(output_json_filename, encoding='utf-8', mode='w+') as outfile:
            json.dump(output_json, outfile)
        outfile.close()


parser = argparse.ArgumentParser(description='Create json files to be used in IE')
parser.add_argument('-input_folder',type=str,
                    help='input folder containing the train/valid/test json')
parser.add_argument('-output_folder',type=str,
                    help='output folder containing the train/valid/test ie json')
parser.add_argument('-type', type=str, default='train',
                    choices=['train', 'valid', 'test'],
                    help='Type of dataset to generate. Options [train|valid|test]')
parser.add_argument('-scoring', action='store_true', help='Generate the json for scoring plays')
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
type = args.type
scoring = args.scoring
process(input_folder, output_folder, type, scoring)
