from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
from melddata import MELDdata
from moseidata import MoseiData
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report
import sys

def run_tone_analyzer(dataset_name, n_texts=None, empty_pred=True):
    """
    Runs IBM Tone Analyzer on the MELD dataset
    Returns ground truth labels as well as the tone analyzer results with the accuracy
    """
    if dataset_name != 'meld' and dataset_name != 'mosei':
        print('Please select the MELD or Mosei dataset')
        sys.exit()

    #Input tone analyzer API Key and URL 
    authenticator = IAMAuthenticator('9_P6Y_TvZJSTAIvYEfP1aKznzP-NDRx-BJtAgsLkBbQN')
    tone_analyzer = ToneAnalyzerV3(
        version= '2017-09-21',
        authenticator=authenticator
    )

    tone_analyzer.set_service_url('https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/7246c9fc-d822-4b82-9b13-7e4c97635f46')

    ###MELD
    if dataset_name == 'meld':
        dataset = MELDdata()
        dataset_path = ""
        dataset.add_ratings_from_files(dataset_path=dataset_path)
    
        dataset.sanitize_all_strings()
        dataset.classify_by_class7()
        #emotions_7 = ['fear', 'sadness', 'anger', 'joy', 'surprise', 'neutral', 'disgust']
        out_emotions = [4, 5, 6]
        for i in out_emotions:
            dataset = dataset.filter_out_class(i)

        dataset,_ = dataset.filter_duplicate_data()

        if n_texts != None:
            dataset,_ = dataset.get_only_n_texts_per_class(100)

        sentences = dataset.get_all_strings()

        emotion_labels = dataset.get_all_labels()
        
    ###MOSEI
    else:
        dataset = MoseiData()
        dataset_path = ""
        dataset.add_ratings_from_files(dataset_path=dataset_path)

        dataset.classify_by_class7()

        emotions_7 = ['fear', 'sadness', 'angry', 'happiness', 'disgust', 'surprise', 'no-emotion']
        out_emotions = [4, 5, 6]
        for i in out_emotions:
            dataset = dataset.filter_out_class(i)
    
        if n_texts != None:
            dataset,_ = dataset.get_only_n_texts_per_class(100)

        sentences = dataset.get_all_strings()

        emotion_labels = dataset.get_all_labels()    

    #Analyze text
    for reviews in sentences:
        tone_analysis = tone_analyzer.tone(
            {'text':reviews},
            content_type='application/json',
            sentences = False
        ).get_result()
        #Write results to files
        jsonText = json.dumps(tone_analysis, indent=2)
        outfile = open('dataset_files/' + str(dataset_name) + '_' + str(sentences.index(reviews)) + '.json', 'w')
        outfile.write(jsonText)
        outfile.close()
    
    tone_analyzer_results = {}
    #tone_analyzer_emotions = ['anger', 'fear', 'sadness', 'joy', 'analytical', 'confident', 'tentative']
    #filter out uncomparable emotions
    tone_analyzer_emotions = ['fear', 'sadness', 'anger', 'joy'] #mosei order
    for i in range(len(emotion_labels)):
        tone_analyzer_results[i] = []
        #Read output files
        with open('dataset_files/' + str(dataset_name) + '_' + str(i) + '.json') as infile:
            data = json.load(infile)
            for tone in data['document_tone']['tones']:
                if tone['tone_id'] in tone_analyzer_emotions:
                    tone_analyzer_results[i].append(tone_analyzer_emotions.index(tone['tone_id']))

    y_pred = []
    for i in range(len(emotion_labels)):
        y_pred.append(tone_analyzer_results[i])
    
    #Accuracy
    if dataset_name == 'meld':
        #Include Empty Sets
        if empty_pred:
            total_true_positives = 0
            for j in range(len(y_pred)):
                if emotion_labels[j] in y_pred[j]:
                    total_true_positives += 1
        
            total_pred = 0
            for k in range(len(y_pred)):
                if len(y_pred[k]) == 0:
                    total_pred += 1
                else:
                    length = len(y_pred[k])
                    total_pred += length
                    
            anger_count = 0
            fear_count = 0
            sadness_count = 0
            joy_count = 0
            for index in range(len(emotion_labels)):
                if emotion_labels[index] == 0:
                    if emotion_labels[index] in y_pred[index]:
                        fear_count += 1
                elif emotion_labels[index] == 1:
                    if emotion_labels[index] in y_pred[index]:
                        sadness_count += 1
                elif emotion_labels[index] == 2:
                    if emotion_labels[index] in y_pred[index]:
                        anger_count += 1
                else:
                    if emotion_labels[index] in y_pred[index]:
                        joy_count += 1

            anger_accuracy = round(anger_count/emotion_labels.count(2), 2)
            fear_accuracy = round(fear_count/emotion_labels.count(0), 2)
            sadness_accuracy = round(sadness_count/emotion_labels.count(1), 2)
            joy_accuracy = round(joy_count/emotion_labels.count(3), 2)
                        
        #Exclude Empty Sets
        else:
            total_true_positives = 0
            for j in range(len(y_pred)):
                if len(y_pred[j]) > 0:
                    if emotion_labels[j] in y_pred[j]:
                        total_true_positives += 1

            total_pred = 0
            for k in range(len(y_pred)):
                if len(y_pred[k]) > 0:
                    length = len(y_pred[k])
                    total_pred += length

            anger_not_empty = 0
            fear_not_empty = 0
            sadness_not_empty = 0
            joy_not_empty = 0
            anger_total = 0
            fear_total = 0
            sadness_total = 0
            joy_total = 0
            for index in range(len(emotion_labels)):
                if emotion_labels[index] == 0:
                    if len(y_pred[index]) > 0:
                        fear_total += 1
                        if emotion_labels[index] in y_pred[index]:
                            fear_not_empty += 1
                elif emotion_labels[index] == 1:
                    if len(y_pred[index]) > 0:
                        sadness_total += 1
                        if emotion_labels[index] in y_pred[index]:
                            sadness_not_empty += 1
                elif emotion_labels[index] == 2:
                    if len(y_pred[index]) > 0:
                        anger_total += 1
                        if emotion_labels[index] in y_pred[index]:
                            anger_not_empty += 1
                else:
                    if len(y_pred[index]) > 0:
                        joy_total += 1
                        if emotion_labels[index] in y_pred[index]:
                            joy_not_empty += 1

            anger_accuracy = round(anger_not_empty/anger_total, 2)
            fear_accuracy = round(fear_not_empty/fear_total, 2)
            sadness_accuracy = round(sadness_not_empty/sadness_total, 2)
            joy_accuracy = round(joy_not_empty/joy_total, 2)

        percent_missed = round(y_pred.count([])/len(y_pred))
        accuracy = round(total_true_positives/total_pred,2)
        return accuracy, anger_accuracy, fear_accuracy, sadness_accuracy, joy_accuracy, percent_missed          

    #Mosei Metrics
    else:
        
        match_list = dataset.get_all_emotion_matches(y_pred, empty_pred)
        complete_match_list = dataset.get_all_complete_emotion_matches(y_pred, empty_pred)
        
        if empty_pred:
            #Match accuracies
            fear_count = 0
            sadness_count = 0
            anger_count = 0
            happiness_count = 0
            for i in range(len(emotion_labels)):
                if emotion_labels[i] == 0:
                    if match_list[i]:
                        fear_count += 1
                elif emotion_labels[i] == 1:
                    if match_list[i]:
                        sadness_count += 1
                elif emotion_labels[i] == 2:
                    if match_list[i]:
                        anger_count += 1
                else:
                    if match_list[i]:
                        happiness_count += 1
                        
            match_accuracy = round(match_list.count(True)/len(match_list), 2)
            fear_match_accuracy = round(fear_count/emotion_labels.count(0), 2)
            sadness_match_accuracy = round(sadness_count/emotion_labels.count(1), 2)
            anger_match_accuracy = round(anger_count/emotion_labels.count(2), 2)
            happiness_match_accuracy = round(happiness_count/emotion_labels.count(3), 2)

            #Complete match accuracies
            fear_comp_count = 0
            sadness_comp_count = 0
            anger_comp_count = 0
            happiness_comp_count = 0
            for i in range(len(complete_match_list)):
                if emotion_labels[i] == 0:
                    if complete_match_list[i]:
                        fear_comp_count += 1
                elif emotion_labels[i] == 1:
                    if complete_match_list[i]:
                        sadness_comp_count += 1
                elif emotion_labels[i] == 2:
                    if complete_match_list[i]:
                        anger_comp_count += 1
                else:
                    if complete_match_list[i]:
                        happiness_comp_count += 1

            complete_match_accuracy = round(complete_match_list.count(True)/len(complete_match_list), 2)
            fear_comp_match_accuracy = round(fear_comp_count/emotion_labels.count(0), 2)
            sadness_comp_match_accuracy = round(sadness_comp_count/emotion_labels.count(1), 2)
            anger_comp_match_accuracy = round(anger_comp_count/emotion_labels.count(2), 2)
            happiness_comp_match_accuracy = round(happiness_comp_count/emotion_labels.count(3), 2)
            percent_missed = round(y_pred.count([])/len(y_pred),2)
            
        #Exclude empty predictions
        else:
            #Match Accuracies
            fear_match_count = 0
            sadness_match_count = 0
            anger_match_count = 0
            happiness_match_count = 0

            fear_comp_count = 0
            sadness_comp_count = 0
            anger_comp_count = 0
            happiness_comp_count = 0
            
            fear_not_empty = 0
            sadness_not_empty = 0
            anger_not_empty = 0
            happiness_not_empty = 0
            
            index_count = 0
            for index in range(len(emotion_labels)):
                if emotion_labels[index] == 0:
                    if len(y_pred[index]) > 0:
                        fear_not_empty += 1
                        if match_list[index_count]:
                            fear_match_count += 1
                        if complete_match_list[index_count]:
                            fear_comp_count += 1
                        index_count += 1
                        
                elif emotion_labels[index] == 1:
                    if len(y_pred[index]) > 0:
                        sadness_not_empty += 1
                        if match_list[index_count]:
                            sadness_match_count += 1
                        if complete_match_list[index_count]:
                            sadness_comp_count += 1
                        index_count += 1
                        
                elif emotion_labels[index] == 2:
                    if len(y_pred[index]) > 0:
                        anger_not_empty += 1
                        if match_list[index_count]:
                            anger_match_count += 1
                        if complete_match_list[index_count]:
                            anger_comp_count += 1
                        index_count += 1
                else:
                    if len(y_pred[index]) > 0:
                        happiness_not_empty += 1
                        if match_list[index_count]:
                            happiness_match_count += 1
                        if complete_match_list[index_count]:
                            happiness_comp_count += 1
                        index_count += 1

            match_accuracy = round(match_list.count(True)/len(match_list), 2)
            complete_match_accuracy = round(complete_match_list.count(True)/len(complete_match_list), 2)
            fear_comp_match_accuracy = round(fear_comp_count/fear_not_empty, 2)
            sadness_comp_match_accuracy = round(sadness_comp_count/sadness_not_empty, 2)
            anger_comp_match_accuracy = round(anger_comp_count/anger_not_empty, 2)
            happiness_comp_match_accuracy = round(happiness_comp_count/happiness_not_empty, 2)
            fear_match_accuracy = round(fear_match_count/fear_not_empty, 2)
            sadness_match_accuracy = round(sadness_match_count/sadness_not_empty, 2)
            anger_match_accuracy = round(anger_match_count/anger_not_empty, 2)
            happiness_match_accuracy = round(happiness_match_count/happiness_not_empty, 2)           
            percent_missed = round(y_pred.count([])/len(y_pred),2)
            
        return (match_accuracy, fear_match_accuracy, sadness_match_accuracy, anger_match_accuracy, happiness_match_accuracy,
                complete_match_accuracy, fear_comp_match_accuracy, sadness_comp_match_accuracy, anger_comp_match_accuracy, happiness_comp_match_accuracy,
                percent_missed)

