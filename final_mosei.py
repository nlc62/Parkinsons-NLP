import os
import csv
import sys

#Extract sentences
text_dict = {}
path = "Transcript/Segmented/Combined/"
#path = "Transcript/Full/Alignment/P2FA/Combined"
filenames = sorted(os.listdir(path))

for text in filenames:
    text_path = path + text
    identifier = text[:-4]

    with open(text_path) as text_file:
        sentences = text_file.read()
        sentences = sentences.split('\n')
        sentences = sentences[:-1]
        
        for sentence in sentences:
            sentence_number_before = sentence.find('___') + 2
            sentence_number_after = sentence.find('___', sentence_number_before)
            sentence_number = sentence[sentence_number_before + 1: sentence_number_after]
            index = sentence.rfind('_')
            text_dict[identifier + '_' + sentence_number] = sentence[index+1:]

labels = []
label_dict = {}
#keys: id_#, values: [[sentiment], [emotions]]
label_dict_keys = []
with open("Labels/mturk_extra_v2.csv") as infile:
    labels = list(csv.reader(infile))
    labels = labels[1:]
    
    for label in labels:
        
        if (label[27] + '_' + label[28]) not in label_dict_keys:
            label_dict_keys.append(label[27] + '_' + label[28])
            label_dict[label[27] + '_' + label[28]] = [[], [[], [], [], [], [], []]]

        #sentiment labels
        label_dict[label[27] + '_' + label[28]][0].append(int(label[38]))
        
        #emotion labels
        #29:anger, 30:disgust, 31:fear, 33:happiness, 34:sadness, 39:surprise
        #31:fear, 34:sadness, 29:angry, 33:happiness, 30:disgust, 39:surprise
                
        label_dict[label[27] + '_' + label[28]][1][0].append(int(label[31]))
        label_dict[label[27] + '_' + label[28]][1][1].append(int(label[34]))
        label_dict[label[27] + '_' + label[28]][1][2].append(int(label[29]))
        label_dict[label[27] + '_' + label[28]][1][3].append(int(label[33]))
        label_dict[label[27] + '_' + label[28]][1][4].append(int(label[30]))
        label_dict[label[27] + '_' + label[28]][1][5].append(int(label[39]))

final_label_dict = {}
for key in label_dict.keys():
    
    final_label_dict[key] = [0, []]
    final_label_dict[key][0] = (sum(label_dict[key][0]))/(len(label_dict[key][0]))
    for emotion in range(6):
        #Set average emotion threshold to 0.5
        emotion_magnitude = sum(label_dict[key][1][emotion])/len(label_dict[key][1][emotion])
        
        final_label_dict[key][1].append(emotion_magnitude)

#Add neutral class for no emotional ground truth
for labels in final_label_dict.keys():
    if final_label_dict[labels][1].count(0) == 6:
        final_label_dict[labels][1].append(1.0)
    else:
        final_label_dict[labels][1].append(0.0)
        
#Delete keys not in text_dict
del_keys = []
for k in final_label_dict.keys():
    if k not in text_dict.keys():
        del_keys.append(k)

for del_key in del_keys:
    del final_label_dict[del_key]

print(final_label_dict)
