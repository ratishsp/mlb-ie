import sys
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from text2num import text2num, NumberException

full_names = ['Baltimore Orioles', 'LA Dodgers', 'Arizona D-backs', 'Texas Rangers',
              'San Diego Padres', 'Tampa Bay Rays', 'Toronto Blue Jays', 'Houston Astros',
              'NY Mets', 'Boston Red Sox', 'Chicago White Sox', 'Milwaukee Brewers',
              'LA Angels', 'Cleveland Indians', 'Florida Marlins', 'Kansas City Royals',
              'St. Louis Cardinals', 'Seattle Mariners', 'Philadelphia Phillies', 'Colorado Rockies',
              'Atlanta Braves', 'San Francisco Giants', 'NY Yankees', 'Oakland Athletics',
              'Washington Nationals', 'Miami Marlins', 'Cincinnati Reds', 'Pittsburgh Pirates',
              'Minnesota Twins', 'Detroit Tigers', 'Chicago Cubs']
'''
full_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
 'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers',
 'Miami Heat', 'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
 'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards', 'Dallas Mavericks',
 'Denver Nuggets', 'Golden State Warriors', 'Houston Rockets', 'Los Angeles Clippers',
 'Los Angeles Lakers', 'Memphis Grizzlies', 'Minnesota Timberwolves', 'New Orleans Pelicans',
 'Oklahoma City Thunder', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings',
 'San Antonio Spurs', 'Utah Jazz']
'''
cities, teams = set(), set()
ec = {} # equivalence classes
for team in full_names:
    pieces = team.split()
    if len(pieces) == 2:
        ec[team] = [pieces[0], pieces[1]]
        cities.add(pieces[0])
        teams.add(pieces[1])
    elif pieces[0] in ["Toronto", "Boston", "Chicago"] : # only 2-word team
        ec[team] = [pieces[0], " ".join(pieces[1:])]
        cities.add(pieces[0])
        teams.add(" ".join(pieces[1:]))
    else: # must be a 2-word City
        ec[team] = [" ".join(pieces[:2]), pieces[2]]
        cities.add(" ".join(pieces[:2]))
        teams.add(pieces[2])

def same_ent(e1, e2):
    if e1 in cities or e1 in teams or e2 in cities or e2 in teams:
        return e1 == e2 or any((e1 in fullname and e2 in fullname for fullname in full_names))
    else:
        return e1 in e2 or e2 in e1

def int_value(input):
    a_number = False
    try: 
        value = int(input)
        a_number = True
    except ValueError:
        pass

    if not a_number:
            value = text2num(input)
    return value


def trip_match(t1, t2):
    try:
        match_found = int_value(t1[1]) == int_value(t2[1]) and t1[2] == t2[2] and same_ent(t1[0], t2[0])
    except NumberException:  # value is double, home run etc
        match_found = False
        if t1[2].startswith('P-BY-P-') and t1[2] == t2[2] and same_ent(t1[0], t2[0]):
            match_found = True
        elif t1[1] == t2[1] and t1[2] == t2[2] and same_ent(t1[0], t2[0]):
            match_found = True
    return match_found

def dedup_triples(triplist):
    """
    this will be inefficient but who cares
    """
    dups = set()
    for i in range(1, len(triplist)):
        for j in range(i):
            if trip_match(triplist[i], triplist[j]):
                dups.add(i)
                break
    return [thing for i, thing in enumerate(triplist) if i not in dups]

def get_triples(fi):
    all_triples = []
    curr = []
    with open(fi) as f:
        for line in f:
            if line.isspace():
                all_triples.append(curr)
                curr = []
            else:
                pieces = line.strip().split('|')
                curr.append(tuple(pieces))
    if len(curr) > 0:
        all_triples.append(curr)
    return all_triples

def calc_precrec(goldfi, predfi):
    gold_triples = get_triples(goldfi)
    pred_triples = get_triples(predfi)
    total_tp, total_predicted, total_gold = 0, 0, 0
    assert len(gold_triples) == len(pred_triples)
    for i, triplist in enumerate(pred_triples):
        tp = 0
        corresponding_gold_triples = list(gold_triples[i])
        for j in range(len(triplist)):
            match_index = -1
            for k in range(len(corresponding_gold_triples)):
                if trip_match(triplist[j], corresponding_gold_triples[k]):
                    match_index = k
                    tp += 1
                    break
            if match_index != -1:
                del corresponding_gold_triples[match_index]
        total_tp += tp
        total_predicted += len(triplist)
        total_gold += len(gold_triples[i])
    avg_prec = float(total_tp)/total_predicted
    avg_rec = float(total_tp)/total_gold
    print("totals:", total_tp, total_predicted, total_gold)
    print("prec:", avg_prec, "rec:", avg_rec)
    return avg_prec, avg_rec

def norm_dld(l1, l2):
    ascii_start = 0
    # make a string for l1
    # all triples are unique...
    s1 = ''.join((chr(ascii_start+i) for i in range(len(l1))))
    s1_upd = list(s1)
    for i in range(len(l1)):
        for j in range(i+1, len(l1)):
            if trip_match(l1[i], l1[j]):
                s1_upd[j] = s1[i]
    s1_upd = ''.join(s1_upd)
    s2 = ''
    next_char = ascii_start + len(s1)
    for j in range(len(l2)):
        found = None
        #next_char = chr(ascii_start+len(s1)+j)
        for k in range(len(l1)):
            if trip_match(l2[j], l1[k]):
                found = s1_upd[k]
                #next_char = s1[k]
                break
        if found is None:
            s2 += chr(next_char)
            next_char += 1
            #assert next_char <= 128
        else:
            s2 += found
    # return 1- , since this thing gives 0 to perfect matches etc
    return 1.0-normalized_damerau_levenshtein_distance(s1_upd, s2)

def calc_dld(goldfi, predfi):
    gold_triples = get_triples(goldfi)
    pred_triples = get_triples(predfi)
    assert len(gold_triples) == len(pred_triples)
    total_score = 0
    for i, triplist in enumerate(pred_triples):
        total_score += norm_dld(triplist, gold_triples[i])
    avg_score = float(total_score)/len(pred_triples)
    print("avg score:", avg_score)
    return avg_score

calc_precrec(sys.argv[1], sys.argv[2])
calc_dld(sys.argv[1], sys.argv[2])

# usage python non_rg_metrics.py gold_tuple_fi pred_tuple_fi
