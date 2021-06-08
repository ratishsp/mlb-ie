import codecs, json, os
from collections import Counter, OrderedDict
from nltk import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import h5py
import random
import math
from text2num import text2num, NumberException
import argparse

random.seed(2)


number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self,k):
        self[k] = []
        return self[k]



def get_team_information(thing, home):
    teams = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"

    teams.add(thing[team_type + "name"])
    teams.add(" ".join([thing[team_type + "city"], thing[team_type + "name"]]))

    alternate_names = {"D-backs": "Diamondbacks", "Diamondbacks": "D-backs", "Athletics": "A 's"}
    for key in alternate_names:
        if thing[team_type + "name"] == key:
            teams.add(" ".join([thing[team_type + "city"], alternate_names[key]]))
            teams.add(alternate_names[key])
    return teams


def get_city_information(thing, home):
    cities = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"
    cities.add(thing[team_type + "city"])

    alternate_names = {"Chi Cubs": ["Chicago"], "LA Angels": ["Los Angeles", "LA"], "LA Dodgers": ["Los Angeles", "LA"],
                       "NY Yankees": ["New York", "NY"], "NY Mets": ["New York", "NY"], "Chi White Sox": ["Chicago"]}
    for key in alternate_names:
        if thing[team_type + "city"] == key:
            for val in alternate_names[key]:
                cities.add(val)
    return cities


def get_ents(thing):
    players = set()
    teams = set()
    cities = set()
    teams.update(get_team_information(thing, home=False))
    teams.update(get_team_information(thing, home=True))
    cities.update(get_city_information(thing, home=False))
    cities.update(get_city_information(thing, home=True))
    players.update(thing["box_score"]["full_name"].values())
    players.update(thing["box_score"]["last_name"].values())
    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            for piece_index in range(len(pieces)):
                entset.add(" ".join(pieces[:piece_index]))
    all_ents = players | teams | cities
    return all_ents, players, teams, cities


def extract_entities(sent, all_ents, prons, prev_ents=None, resolve_prons=False,
        players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in all_ents: # findest longest spans; only works if we put in words...
            j = 1
            while i+j <= len(sent) and " ".join(sent[i:i+j]) in all_ents:
                j += 1
            sent_ents.append((i, i+j-1, " ".join(sent[i:i+j-1]), False))
            i += j-1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = set(["three point", "three - point", "three - pt", "three pt", "three - pointers", "three - pointer", "three pointers"])
    return " ".join(sent[i:i+3]) in ignores or " ".join(sent[i:i+2]) in ignores


def extract_numbers(sent):
    sent_nums = []
    i = 0
    #print sent
    while i < len(sent):
        toke = sent[i]
        a_number = False
        to_evaluate = toke.replace("/","")  # handle 1/3
        try:
            itoke = float(to_evaluate)
            a_number = True
        except ValueError:
            pass
        if a_number:
            sent_nums.append((i, i+1, toke))
            i += 1
        elif toke in number_words: # and not annoying_number_word(sent, i): # get longest span  (this is kind of stupid)
            j = 1
            while i+j < len(sent) and sent[i+j] in number_words: # and not annoying_number_word(sent, i+j):
                j += 1
            try:
                sent_nums.append((i, i+j, text2num(" ".join(sent[i:i+j]))))
            except NumberException:
                pass
                #print sent
                #print sent[i:i+j]
                #assert False
            i += j
        else:
            i += 1
    return sent_nums


#actions such as single, double, homer
def extract_intransitive_actions(sent):
    int_actions = []
    two_word = set(["home run", "home runs", "grounded out", "ground out", "flied out", "sacrifice fly"])
    consider = set(["single", "double", "doubles", "homer", "homers", "scored", "error", "errors", "singled", "doubled",
                    "homered", "singles", "triple", "triples", "tripled", "walk", "walks", "walked", "groundout", "RBI"])
    toke_action_dict = {"single": "single", "double": "double", "doubles": "double", "homer": "home_run",
                        "homers": "home_run", "home run": "home_run", "scored": "scorer", "error": "fielder_error",
                        "errors": "fielder_error", "singled": "single", "doubled": "double", "homered": "home_run",
                        "home runs": "home_run", "singles": "single", "triple": "triple", "triples": "triple",
                        "tripled": "triple", "walk":"walk", "walks":"walk", "walked":"walk", "groundout": "groundout",
                        "grounded out":"groundout", "ground out":"groundout", "flied out": "flyout",
                        "sacrifice fly": "sac_fly", "RBI": "rbi"}
    for i in range(len(sent)):
        toke = sent[i]
        if toke in consider:
            int_actions.append((i, i+1, toke_action_dict[toke]))
        elif " ".join(sent[i: i+2]) in two_word:
            int_actions.append((i, i+2, toke_action_dict[" ".join(sent[i: i+2])]))
    return int_actions


def get_player_idx(bs, entname, names_map):
    keys = []
    for k, v in bs["full_name"].iteritems():
         if entname == v:
             keys.append(k)
             names_map[bs["last_name"][k]] = k
    if len(keys) == 0:
        if entname in names_map:
            keys.append(names_map[entname])
    if len(keys) == 0:
        for k,v in bs["last_name"].iteritems():
            if entname == v:
                keys.append(k)
                names_map[entname] = k
        if len(keys) > 1: # take the earliest one
            keys.sort(key = lambda x: int(x))
            keys = keys[:1]
            names_map[entname] = keys[0]
    if len(keys) == 0:
        for k,v in bs["first_name"].iteritems():
            if entname == v:
                keys.append(k)
        if len(keys) > 1: # if we matched on first name and there are a bunch just forget about it
            return None
    #assert len(keys) <= 1, entname + " : " + str(bs["full_name"].values())
    return keys[0] if len(keys) > 0 else None


def get_inning(sent, prev_sent_context, ordinal_adjective_map):
    inning_identifier = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                         "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th"}
    inning_identifier_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                             "eighth": 8, "ninth": 9, "tenth": 10, "7th": 7, "8th": 8, "9th": 9, "10th": 10, "11th": 11,
                             "12th": 12, "13th": 13, "14th": 14, "15th": 15}
    additional_check = {"16th", "17th", "18th", "19th", "20th", "21st", "22nd", "23rd", "24th", "25th", "26th", "27th",
                        "28th", "29th", "30th"}
    stops = stopwords.words('english')
    innings = []
    upd_sent = " ".join(sent)
    upd_sent = upd_sent.replace("-", " ").split()  # handles cases such as these: pitched out of a second-inning jam
    intersected = set(upd_sent).intersection(inning_identifier)
    if len(intersected) > 0:
        # candidate present
        for i in range(len(sent)):
            if sent[i] in inning_identifier and i+1 < len(sent) and sent[i+1] in ["inning", "innings"]:
                innings.append((inning_identifier_map[sent[i]], i))
            elif "-" in sent[i]  and sent[i].split("-")[0] in inning_identifier and sent[i].split("-")[1]  == "inning":
                innings.append((inning_identifier_map[sent[i].split("-")[0]], i))
            elif (" ".join(sent[:i]).endswith("in the") or " ".join(sent[:i]).endswith("in the top of the") or " ".join(
                    sent[:i]).endswith("in the bottom of the")) and sent[i] in inning_identifier and (
                    (i + 1 < len(sent) and (sent[i + 1] in [".", ","] or sent[i + 1] in stops)) or i + 1 == len(sent)):
                innings.append((inning_identifier_map[sent[i]], i))
            elif sent[i] in inning_identifier and ((i+1 < len(sent) and (sent[i+1] in [".", ","] or sent[i+1] in stops)) or i+1 == len(sent)):
                # i+1 == len(sent) handles the case such as "Kapler also doubled in a run in the first "; no full stop at the end
                expanded_context = prev_sent_context + sent[:i+1]
                expanded_context = " ".join(expanded_context)
                assert expanded_context in ordinal_adjective_map
                if ordinal_adjective_map[expanded_context] == "True":
                    innings.append((inning_identifier_map[sent[i]], i))
    return innings


def get_rels(entry, ents, nums, int_actions, players, teams, cities, tokes, innings, names_map):
    """
    this looks at the box/line score and figures out which (entity, number) pairs
    are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score together),
    we give a NONE label, so for generated summaries that we extract from, if we predict
    a label we'll get it wrong (which is presumably what we want).
    N.B. this function only looks at the entity string (not position in sentence), so the
    string a pronoun corefers with can be snuck in....
    """
    rels = []
    bs = entry["box_score"]
    for i, ent in enumerate(ents):
        if ent[3]: # pronoun
            continue # for now
        entname = ent[2]
        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities and entname not in teams:
            pidx = get_player_idx(bs, entname, names_map)
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if pidx is not None: # player might not actually be in the game or whatever
                    for colname, col in bs.iteritems():
                        if pidx in col and col[pidx] == strnum: # allow multiple for now
                            if len(tokes) > numtup[1] and tokes[numtup[1]] == "outs" or (len(tokes) > numtup[1] + 1 and tokes[numtup[1]] == "-"  and tokes[numtup[1]+1] == "out"):  # ignore two outs or two - out single
                                continue
                            if colname in ["ab", "bb", "hr", "so", "e", "po", "go", "ao", "lob", "d", "r", "cs", "sf", "sac", "t", "hbp", "fldg", "p_hr", "rbi"]:
                                continue
                            if colname in ["p_bs", "p_sv", "p_hld"] and strnum == "0":
                                continue
                            if len(ents)> i+1 and ent[0] < ents[i+1][0] < numtup[0]:  # if there is another entity in between the current entity and num tuple, ignore
                                continue
                            if i > 0 and numtup[0] < ents[i-1][0] < ent[0]:  # if there is another entity in between the current entity and num tuple
                                # and the order is numtuple ent0 ent1, ignore # check for non pronoun
                                continue
                            if colname == "h" and len(tokes) > numtup[1] and tokes[numtup[1]] != "hits":
                                continue
                            if colname == "sb" and len(tokes) > numtup[1] and tokes[numtup[1]] != "stolen":
                                continue
                            if colname == "a" and len(tokes) > numtup[1] and tokes[numtup[1]] not in ["assists", "assist"] and (len(tokes) <= numtup[1] + 1 or tokes[numtup[1]+1] not in ["assists", "assist"]):
                                continue
                            rels.append((ent, numtup, "PLAYER-" + colname, pidx))
                            found = True
                    if innings:
                        for colname in ["o", "runs"]:
                            for inning in innings:
                                plays = entry["play_by_play"]
                                if str(inning[0]) not in plays:
                                    continue
                                for top_bottom in ["top", "bottom"]:
                                    inning_plays = plays[str(inning[0])][top_bottom]
                                    for inning_play in inning_plays:
                                        if colname == "o" and colname in inning_play and str(inning_play[
                                            colname]) == strnum and len(tokes) > numtup[1] + 1 and tokes[
                                            numtup[1]] == "-" and tokes[
                                            numtup[1] + 1] == "out":
                                            if "batter" in inning_play and inning_play[
                                                "batter"] == bs["full_name"][pidx]:
                                                rels.append((ent, numtup, "P-BY-P-" + colname, pidx))
                                                found = True
                                            elif "pitcher" in inning_play and inning_play[
                                                "pitcher"] == bs["full_name"][pidx]:
                                                rels.append((ent, numtup, "P-BY-P-" + colname + "_pitcher", pidx))
                                                found = True
                                        elif colname == "runs" and colname in inning_play and str(inning_play[
                                            colname]) == strnum and len(tokes) > numtup[1] + 1 and tokes[
                                            numtup[1]] == "-" and tokes[
                                            numtup[1] + 1] == "run":
                                            if "batter" in inning_play and inning_play[
                                                "batter"] == bs["full_name"][pidx]:
                                                rels.append((ent, numtup, "P-BY-P-" + colname, pidx))
                                                found = True
                                            elif "pitcher" in inning_play and inning_play[
                                                "pitcher"] == bs["full_name"][pidx]:
                                                rels.append((ent, numtup, "P-BY-P-" + colname + "_pitcher", pidx))
                                                found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None))
            for j, inttup in enumerate(int_actions):
                found = False
                if pidx is not None:
                    if inttup[2] in ["single", "double", "triple", "home_run", "scorer", "fielder_error", "walk", "groundout", "flyout", "sac_fly", "rbi"] :
                        found = check_batter_fielder_in_inning(ent, bs["full_name"][pidx], entry, innings, inttup, pidx, rels)
                        if not found:
                            found = check_pitcher_in_inning(ent, bs["full_name"][pidx], entry, innings, inttup, pidx, rels)
                if not found and innings:
                    rels.append((ent, inttup, "NONE", None))
        else: # has to be city or team
            entpieces = entname.split()
            linescore = None
            is_home = None
            if entpieces[-1] == "Sox" and " ".join(entpieces[-2:]) in entry["home_name"]:
                linescore = entry["home_line"]
                is_home = True
            elif entpieces[-1] == "Sox" and " ".join(entpieces[-2:]) in entry["vis_name"]:
                linescore = entry["vis_line"]
                is_home = False
            elif entpieces[0] in entry["home_city"] or entpieces[-1] in entry["home_name"]:
                linescore = entry["home_line"]
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry["vis_name"]:
                linescore = entry["vis_line"]
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    linescore = entry["home_line"]
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    linescore = entry["vis_line"]
                    is_home = False
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if linescore is not None:
                    for colname, val in linescore.iteritems():
                        if colname == "team_errors" and "errors" not in tokes:
                            continue
                        if str(val) == strnum:
                            rels.append((ent, numtup, colname, is_home))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None)) # should i specialize the NONE labels too?
    rels.sort(key=lambda rel: rel[1][0])
    return rels


def check_pitcher_in_inning(ent, entname, entry, innings, inttup, pidx, rels):
    found = False
    if innings:
        plays = entry["play_by_play"]
        for inning in innings:
            if str(inning[0]) not in plays:
                continue
            for top_bottom in ["top", "bottom"]:
                inning_plays = plays[str(inning[0])][top_bottom]
                for inning_play in inning_plays:
                    if "pitcher" in inning_play and inning_play["pitcher"] == entname:
                        for event_candidate in ["single", "double", "triple", "home_run", "walk", "intent_walk", "groundout", "flyout", "sac_fly", "rbi"]:
                            if inning_play["event"].lower().replace(" ", "_") == event_candidate and inttup[
                                2] == event_candidate:
                                rels.append((ent, inttup, "P-BY-P-" + inttup[2] + "_pitcher", pidx))
                                found = True
                                return found
                            elif event_candidate == "intent_walk" and inning_play["event"].lower().replace(" ", "_") == event_candidate and inttup[
                                2] == "walk":
                                rels.append((ent, inttup, "P-BY-P-" + inttup[2] + "_pitcher", pidx))
                                found = True
                                return found
                            elif event_candidate == "rbi" and inttup[2] == event_candidate and "rbi" in inning_play and int(inning_play["rbi"]) > 0:
                                rels.append((ent, inttup, "P-BY-P-" + inttup[2] + "_pitcher", pidx))
                                found = True
                                return found

    return found


def check_batter_fielder_in_inning(ent, entname, entry, innings, inttup, pidx, rels):
    found = False
    if innings:
        plays = entry["play_by_play"]
        scorers_attrib = "scorers"
        for inning in innings:
            if str(inning[0]) not in plays:
                continue
            for top_bottom in ["top", "bottom"]:
                inning_plays = plays[str(inning[0])][top_bottom]
                for inning_play in inning_plays:
                    if "batter" in inning_play and inning_play["batter"] == entname:
                        for event_candidate in ["single", "double", "triple", "home_run", "walk", "intent_walk", "groundout", "flyout", "sac_fly", "rbi"]:
                            if inning_play["event"].lower().replace(" ", "_") == event_candidate and inttup[
                                2] == event_candidate:
                                rels.append((ent, inttup, "P-BY-P-" + inttup[2], pidx))
                                found = True
                                return found
                            elif event_candidate == "intent_walk" and inning_play["event"].lower().replace(" ", "_") == event_candidate and inttup[
                                2] == "walk":
                                rels.append((ent, inttup, "P-BY-P-" + inttup[2], pidx))
                                found = True
                                return found
                            elif event_candidate == "rbi" and inttup[2] == event_candidate and "rbi" in inning_play and int(inning_play["rbi"]) > 0:
                                rels.append((ent, inttup, "P-BY-P-" + inttup[2], pidx))
                                found = True
                                return found
                    elif "fielder_error" in inning_play and inning_play["fielder_error"] == entname and \
                            inttup[2] == "fielder_error":
                        rels.append((ent, inttup, "P-BY-P-" + inttup[2], pidx))
                        found = True
                        return found
                    elif scorers_attrib in inning_play and len(inning_play[scorers_attrib]) > 0 and \
                            inning_play[scorers_attrib][0] != "N/A" and entname in inning_play[scorers_attrib] and \
                            inttup[2] == "scorer":
                        rels.append((ent, inttup, "P-BY-P-" + inttup[2], pidx))
                        found = True
                        return found
    return found


def get_ordinal_adjective_map_file_name(ordinal_inning_map_folder, mode):
    file_list = os.listdir(ordinal_inning_map_folder)
    for file_name in file_list:
        if mode in file_name:
            return os.path.join(ordinal_inning_map_folder, file_name)
    return None


def get_ordinal_adjective_map(ordinal_inning_map_file):
    ordinal_adjective_map_file = codecs.open(ordinal_inning_map_file, mode="r", encoding="utf-8")
    ordinal_adjective_map_lines = ordinal_adjective_map_file.readlines()
    ordinal_adjective_map_lines = [line.strip() for line in ordinal_adjective_map_lines]
    ordinal_adjective_map = {}
    for line in ordinal_adjective_map_lines:
        ordinal_adjective_map[line.split("\t")[0]] = line.split("\t")[1]
    return ordinal_adjective_map


def append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, candrels, ordinal_adjective_map=None, delimiter=None):
    """
    appends tuples of form (sentence_tokens, [rels]) to candrels
    """
    if delimiter is None:
        delimiter = " <segment> "
    names_map = {}
    sents = summ.split(delimiter)
    for j, sent in enumerate(sents):
        #tokes = word_tokenize(sent)
        prev_segment = [] if j == 0 else sents[j - 1].split()
        innings = get_inning(sent.split(), prev_segment, ordinal_adjective_map)
        for individual_sent in sent_tokenize(sent):
            tokes = individual_sent.split()
            ents = extract_entities(tokes, all_ents, prons)
            nums = extract_numbers(tokes)
            int_actions = extract_intransitive_actions(tokes)

            rels = get_rels(entry, ents, nums, int_actions, players, teams, cities, tokes, innings, names_map)
            if len(rels) > 0:
                candrels.append((tokes, rels))
    return candrels


def get_datasets(path="../boxscore-data/rotowire", train_index=-1, ordinal_inning_map_file= None):
    prons = None
    file_name = get_ordinal_adjective_map_file_name(ordinal_inning_map_file, mode="train")
    ordinal_adjective_map = get_ordinal_adjective_map(file_name)
    nugz = []
    for train_json_index in range(23):
        print("train" + str(train_json_index) + ".json")
        with codecs.open(os.path.join(path, "train"+str(train_json_index)+".json"), "r", "utf-8") as f:
            trdata = json.load(f)
            for i, entry in enumerate(trdata):
                all_ents, players, teams, cities = get_ents(entry)
                summ = " ".join(entry['summary'])
                append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, nugz, ordinal_adjective_map, delimiter=" *NEWPARAGRAPH* ")
    extracted_stuff = []
    extracted_stuff.append(nugz)

    ordinal_adjective_maps = []
    with codecs.open(os.path.join(path, "valid.json"), "r", "utf-8") as f:
        valdata = json.load(f)
        file_name = get_ordinal_adjective_map_file_name(ordinal_inning_map_file, mode="valid")
        ordinal_adjective_map = get_ordinal_adjective_map(file_name)
        ordinal_adjective_maps.append(ordinal_adjective_map)

    with codecs.open(os.path.join(path, "test.json"), "r", "utf-8") as f:
        testdata = json.load(f)
        file_name = get_ordinal_adjective_map_file_name(ordinal_inning_map_file, mode="test")
        ordinal_adjective_map = get_ordinal_adjective_map(file_name)
        ordinal_adjective_maps.append(ordinal_adjective_map)

    datasets = [valdata, testdata]
    for dataset_index, dataset in enumerate(datasets):
        nugz = []
        for i, entry in enumerate(dataset):
            all_ents, players, teams, cities = get_ents(entry)
            summ = " ".join(entry['summary'])
            append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, nugz, ordinal_adjective_maps[dataset_index], delimiter=" *NEWPARAGRAPH* ")

        extracted_stuff.append(nugz)

    return extracted_stuff

def append_to_data(tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
    """
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_ent, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    for rel in tup[1]:
        ent, num, label, idthing = rel
        sents.append(sent)
        lens.append(sentlen)
        ent_dists = [j-ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in xrange(max_len)]
        entdists.append(ent_dists)
        num_dists = [j-num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in xrange(max_len)]
        numdists.append(num_dists)
        labels.append(labeldict[label])


def append_multilabeled_data(tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
    """
    used for val, since we have contradictory labelings...
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    # get all the labels for the same rel
    unique_rels = DefaultListOrderedDict()
    for rel in tup[1]:
        ent, num, label, idthing = rel
        unique_rels[ent, num].append(label)

    for rel, label_list in unique_rels.iteritems():
        ent, num = rel
        sents.append(sent)
        lens.append(sentlen)
        ent_dists = [j-ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in xrange(max_len)]
        entdists.append(ent_dists)
        num_dists = [j-num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in xrange(max_len)]
        numdists.append(num_dists)
        labels.append([labeldict[label] for label in label_list])


def append_labelnums(labels):
    labelnums = [len(labellist) for labellist in labels]
    max_num_labels = max(labelnums)
    print "max num labels", max_num_labels

    # append number of labels to labels
    for i, labellist in enumerate(labels):
        labellist.extend([-1]*(max_num_labels - len(labellist)))
        labellist.append(labelnums[i])

# for full sentence IE training
def save_full_sent_data(outfile, path="../boxscore-data/rotowire", train_index=-1, ordinal_inning_map_file=None, multilabel_train=False, nonedenom=0):
    datasets = get_datasets(path, train_index, ordinal_inning_map_file= ordinal_inning_map_file)
    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets[0]]
    for k in word_counter.keys():
        if word_counter[k] < 2:
            del word_counter[k] # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = dict(((wrd, i+1) for i, wrd in enumerate(word_counter.keys())))
    labelset = set()
    [labelset.update([rel[2] for rel in tup[1]]) for tup in datasets[0]]
    labeldict = dict(((label, i+1) for i, label in enumerate(labelset)))

    # save stuff
    trsents, trlens, trentdists, trnumdists, trlabels = [], [], [], [], []
    valsents, vallens, valentdists, valnumdists, vallabels = [], [], [], [], []
    testsents, testlens, testentdists, testnumdists, testlabels = [], [], [], [], []

    max_trlen = max((len(tup[0]) for tup in datasets[0]))
    print "max tr sentence length:", max_trlen

    # do training data
    for tup in datasets[0]:
        if multilabel_train:
            append_multilabeled_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)
        else:
            append_to_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)

    if multilabel_train:
        append_labelnums(trlabels)

    if nonedenom > 0:
        # don't keep all the NONE labeled things
        none_idxs = [i for i, labellist in enumerate(trlabels) if labellist[0] == labeldict["NONE"]]
        random.shuffle(none_idxs)
        # allow at most 1/(nonedenom+1) of NONE-labeled
        num_to_keep = int(math.floor(float(len(trlabels)-len(none_idxs))/nonedenom))
        print "originally", len(trlabels), "training examples"
        print "keeping", num_to_keep, "NONE-labeled examples"
        ignore_idxs = set(none_idxs[num_to_keep:])

        # get rid of most of the NONE-labeled examples
        trsents = [thing for i,thing in enumerate(trsents) if i not in ignore_idxs]
        trlens = [thing for i,thing in enumerate(trlens) if i not in ignore_idxs]
        trentdists = [thing for i,thing in enumerate(trentdists) if i not in ignore_idxs]
        trnumdists = [thing for i,thing in enumerate(trnumdists) if i not in ignore_idxs]
        trlabels = [thing for i,thing in enumerate(trlabels) if i not in ignore_idxs]

    print len(trsents), "training examples"

    # do val, which we also consider multilabel
    max_vallen = max((len(tup[0]) for tup in datasets[1]))
    for tup in datasets[1]:
        #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
        append_multilabeled_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_vallen)

    append_labelnums(vallabels)

    print len(valsents), "validation examples"

    # do test, which we also consider multilabel
    max_testlen = max((len(tup[0]) for tup in datasets[2]))
    for tup in datasets[2]:
        #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
        append_multilabeled_data(tup, testsents, testlens, testentdists, testnumdists, testlabels, vocab, labeldict, max_testlen)

    append_labelnums(testlabels)

    print len(testsents), "test examples"

    h5fi = h5py.File(outfile, "w")
    h5fi["trsents"] = np.array(trsents, dtype=int)
    h5fi["trlens"] = np.array(trlens, dtype=int)
    h5fi["trentdists"] = np.array(trentdists, dtype=int)
    h5fi["trnumdists"] = np.array(trnumdists, dtype=int)
    h5fi["trlabels"] = np.array(trlabels, dtype=int)
    h5fi["valsents"] = np.array(valsents, dtype=int)
    h5fi["vallens"] = np.array(vallens, dtype=int)
    h5fi["valentdists"] = np.array(valentdists, dtype=int)
    h5fi["valnumdists"] = np.array(valnumdists, dtype=int)
    h5fi["vallabels"] = np.array(vallabels, dtype=int)
    #h5fi.close()

    #h5fi = h5py.File("test-" + outfile, "w")
    h5fi["testsents"] = np.array(testsents, dtype=int)
    h5fi["testlens"] = np.array(testlens, dtype=int)
    h5fi["testentdists"] = np.array(testentdists, dtype=int)
    h5fi["testnumdists"] = np.array(testnumdists, dtype=int)
    h5fi["testlabels"] = np.array(testlabels, dtype=int)
    h5fi.close()
    ## h5fi["vallabelnums"] = np.array(vallabelnums, dtype=int)
    ## h5fi.close()

    # write dicts
    revvocab = dict(((v,k) for k,v in vocab.iteritems()))
    revlabels = dict(((v,k) for k,v in labeldict.iteritems()))
    with codecs.open(outfile.split('.')[0] + ".dict", "w+", "utf-8") as f:
        for i in xrange(1, len(revvocab)+1):
            f.write("%s %d \n" % (revvocab[i], i))

    with codecs.open(outfile.split('.')[0] + ".labels", "w+", "utf-8") as f:
        for i in xrange(1, len(revlabels)+1):
            f.write("%s %d \n" % (revlabels[i], i))


def prep_generated_data(genfile, dict_pfx, outfile, path="../boxscore-data/mlb", test=False, ordinal_inning_map_file=None):
    # recreate vocab and labeldict
    vocab = {}
    with codecs.open(dict_pfx+".dict", "r", "utf-8") as f:
        for line in f:
            pieces = line.strip().split()
            vocab[pieces[0]] = int(pieces[1])

    labeldict = {}
    with codecs.open(dict_pfx+".labels", "r", "utf-8") as f:
        for line in f:
            pieces = line.strip().split()
            labeldict[pieces[0]] = int(pieces[1])

    with codecs.open(genfile, "r", "utf-8") as f:
        gens = f.readlines()

    with codecs.open(os.path.join(path, "valid.json"), "r", "utf-8") as f:
        trdata = json.load(f)

    valfi = "test.json" if test else "valid.json"
    with codecs.open(os.path.join(path, valfi), "r", "utf-8") as f:
        valdata = json.load(f)

    #assert len(valdata) == len(trdata)
    ordinal_adjective_map = get_ordinal_adjective_map(ordinal_inning_map_file)
    nugz = [] # to hold (sentence_tokens, [rels]) tuples
    sent_reset_indices_count = Counter() # sentence indices where a box/story is reset
    sent_reset_indices_count[0] += 1
    prons = None
    for i, entry in enumerate(valdata):
        summ = gens[i]
        all_ents, players, teams, cities = get_ents(entry)
        append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, nugz, ordinal_adjective_map)
        sent_reset_indices_count[len(nugz)]+=1
        #if i == 1:
        #    break

    # save stuff
    max_len = max((len(tup[0]) for tup in nugz))
    psents, plens, pentdists, pnumdists, plabels = [], [], [], [], []

    rel_reset_indices = []
    for t, tup in enumerate(nugz):
        if t in sent_reset_indices_count: # then last rel is the last of its box
            assert len(psents) == len(plabels)
            for index in range(sent_reset_indices_count[t]):
                rel_reset_indices.append(len(psents))
        append_multilabeled_data(tup, psents, plens, pentdists, pnumdists, plabels, vocab, labeldict, max_len)

    append_labelnums(plabels)

    print len(psents), "prediction examples"

    h5fi = h5py.File(outfile, "w")
    h5fi["valsents"] = np.array(psents, dtype=int)
    h5fi["vallens"] = np.array(plens, dtype=int)
    h5fi["valentdists"] = np.array(pentdists, dtype=int)
    h5fi["valnumdists"] = np.array(pnumdists, dtype=int)
    h5fi["vallabels"] = np.array(plabels, dtype=int)
    h5fi["boxrestartidxs"] = np.array(np.array(rel_reset_indices)+1, dtype=int) # 1-indexed
    h5fi.close()

################################################################################


parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-input_path', type=str, default="",
                    help="path to input")
parser.add_argument('-output_fi', type=str, default="",
                    help="desired path to output file")
parser.add_argument('-gen_fi', type=str, default="",
                    help="path to file containing generated summaries")
parser.add_argument('-dict_pfx', type=str, default="roto-ie",
                    help="prefix of .dict and .labels files")
parser.add_argument('-mode', type=str, default='make_ie_data',
                    choices=['make_ie_data', 'prep_gen_data'],
                    help="what utility function to run")
parser.add_argument('-test', action='store_true', help='use test data')
parser.add_argument('-ordinal_inning_map_file', type=str, default="",
                    help="path to file map of ordinal and inning information")
parser.add_argument('-train_index', type=int, default=-1,
                    help="index of training json file")

args = parser.parse_args()

if args.mode == 'make_ie_data':
    save_full_sent_data(args.output_fi, path=args.input_path, train_index=args.train_index, ordinal_inning_map_file=args.ordinal_inning_map_file, multilabel_train=True)
elif args.mode == 'prep_gen_data':
    prep_generated_data(args.gen_fi, args.dict_pfx, args.output_fi, path=args.input_path,
                        test=args.test, ordinal_inning_map_file=args.ordinal_inning_map_file)
