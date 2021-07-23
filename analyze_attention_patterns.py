import numpy as np

skipped = 0
cur_configs = []
semtype2values = {}
with open('knowledge_module_logs.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('text'):
            cur_configs = []
        elif line.startswith('c = {'):
            s_index, e_index = line.index('{'), line.index('}')
            semtypes = eval(line[s_index:e_index+1])['semtypes']
            cur_configs.append(semtypes)
        elif line.startswith('tensor'):
            try:
                s_index, e_index = line.index('['), line.index(']')
            except:
                skipped += 1
                continue
            prob_values = eval(line[s_index:e_index+1])
            for ix in range(len(cur_configs)):
                for semtype in cur_configs[ix]:
                    if not semtype in semtype2values: semtype2values[semtype] = []
                    semtype2values[semtype].append(prob_values[ix])

print('skipped = {}'.format(skipped))
all_semtypes = []
semtype2avgval, semtype2max, semtype2min, semtype2ctx = {}, {}, {}, {}
semtype290percentile = {}
for semtype in semtype2values:
    semtype2ctx[semtype] = len(semtype2values[semtype])
    semtype2avgval[semtype] = round(np.average(semtype2values[semtype]), 3)
    semtype2max[semtype] = round(np.max(semtype2values[semtype]), 3)
    semtype2min[semtype] = round(np.min(semtype2values[semtype]), 3)
    semtype290percentile[semtype] = round(np.percentile(semtype2values[semtype], 90), 3)
    all_semtypes.append(semtype)

all_semtypes = sorted(all_semtypes, key=lambda x: semtype2avgval[x], reverse=True)
for semtype in all_semtypes:
    print('[{}] avg = {} | 90 percentile = {} | max = {} | min = {} | ctx = {}'.format(semtype, semtype2avgval[semtype], semtype290percentile[semtype], semtype2max[semtype], semtype2min[semtype], semtype2ctx[semtype]))
