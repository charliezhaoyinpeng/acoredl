import pickle
import copy

# file = '../ACAR-Net/annotations/ava_train_v2.2_with_fair_0.9.pkl'
# file = '../ACAR-Net/annotations/ava_val_v2.2_fair_0.85.pkl'
file = 'tools/ava_train_v2.2_person_obj.pkl'
import torch

pk_file = pickle.load(open(file, "rb"))
# print(len(pk_file), type(pk_file), pk_file[1])
images = pk_file[0]


# print(pk_file[0])
# print(pk_file[0][0])
# print(len(pk_file[0]))

def method(images):
    ans = []
    for image in images:
        all_person_ids = []
        items = image['labels']
        for item in items:
            person_ids = item['person_id']
            for id in person_ids:
                all_person_ids.append(id)
        if len(all_person_ids) == 0:
            print(image)
            ans.append(image)
        temp = list(set(all_person_ids))
        if len(temp) == 1 and temp[0] == -100:
            print(image)
            ans.append(image)
    # print(inds)
    return ans


print(len(images), len(method(images)))
temp_list = copy.deepcopy(method(images))
# print(temp_list)
for image in temp_list:
    images.remove(image)
print(len(images), len(method(images)))
pk_file[0] = images
save_name = 'ava_train_v2.2_person_obj_ud'
with open('tools/' + save_name + '.pkl', 'wb') as f:
    pickle.dump(pk_file, f)

#         label = item['label']
#         for each_label in label:
#             if each_label < 0 or each_label > 60:
#                 print(each_label, item['person_id'])
#


"""
info = dict()
for image in images:
    video = image['video']
    time = image['time']
    if video in info:
        info[video].append(time)
    else:
        info[video] = []

range = 8
# save_name = 'ava_train_v2.2_person_obj_ud_'
save_name = 'ava_train_v2.2_with_fair_0.9_ud_'

for k, v in info.items():
    mid_ind = int(len(v) / 2 - 1)
    temp_v = v[mid_ind - range: mid_ind + range]
    info[k] = temp_v

up_images = []
for image in images:
    if image['video'] in info.keys() and image['time'] in info[image['video']]:
        up_images.append(image)

print(len(up_images))

pk_file[0] = up_images
with open('tools/' + save_name + str(range) + '.pkl', 'wb') as f:
    pickle.dump(pk_file, f)

"""
