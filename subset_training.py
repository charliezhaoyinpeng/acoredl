import pickle, copy
import pandas as pd

# path = "tools/ava_train_v2.2_person_obj_ud.pkl"
path = "annotations/ava_val_v2.2_fair_0.85.pkl"
df = pd.read_pickle(path)
frames = df[0]
print(df[1])
# print(frames[0])
# for frame in frames:
#     if frame['video'] == 'ZosVdkY76FU' and frame['time'] == 1798:
#         print(frame)




# ans = dict()
# for frame in frames:
#     if frame['video'] not in ans.keys():
#         ans[frame['video']] = [0] * 60
#     actors = frame['labels']
#     for actor in actors:
#         if -100 not in actor['person_id']:
#             for label in actor['label']:
#                 ans[frame['video']][label] += 1
#             # print(actor['person_id'][0], actor['label'])
#
# for k, v in ans.items():
#     print(k, v)

