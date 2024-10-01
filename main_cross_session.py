"""
This is the main script for cross-session paper[1]. 
The code is based on the neurodeckit package.

Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/6/30
License: All rights reserved

[1] Pan, LC, et al. "Cross-session MI-EEG decoding with Riemannian spatial filtering and domain adaptation." 
Chinese Journal of Scientific Instrument (2024).

"""

import os, time
import json
import numpy as np
from neurodeckit import (
    Dataset_Left_Right_MI, 
    Dataset_MI, 
    Pre_Processing, 
    EL_Classifier, 
    TL_Classifier, 
    DL_Classifier, 
    TLSplitter, 
    encode_datasets, 
    )


def models(X_train, y_train, X_test, y_test, target_domain=None, fs=None, file_name=None, njobs=1):
    model_name = []
    clf = []
    ######################################### pre-processing #####################################
    
    # baseline pre-processing
    pre_est = Pre_Processing(fs_new=fs, start_time=0, end_time=4, lowcut=8, highcut=30, cs_method=None)
    
    # pre-processing with RSF method (for TL_Classifier)
    pre_est_rsf = Pre_Processing(fs_new=fs, start_time=0, end_time=4, lowcut=8, highcut=30, cs_method='rsf')
    
    ####################################### Contrast experiment ##################################
    # 1 baseline EA-CSP-LDA
    model_name.append('EA-CSP')
    base_clf1 = TL_Classifier(
        target_domain=target_domain, dpa_method='EA', fee_method='CSP', fes_method=None, 
        clf_method='LDA', end_method=None, ete_method=None, pre_est=pre_est)
    clf.append(base_clf1)
    
    # 2 baseline RA-CSP-LDA
    model_name.append('RA-CSP')
    base_clf2 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method='CSP', fes_method=None, 
        clf_method='LDA', end_method=None, ete_method=None, pre_est=pre_est)
    clf.append(base_clf2)
      
    # 3 baseline RA-TSM-LDA
    model_name.append('TSM')
    base_clf3 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method='TS', fes_method=None, 
        clf_method='LDA', end_method=None, ete_method=None, pre_est=pre_est)
    clf.append(base_clf3)
    
    # 4 baseline RA-MEKT-LDA
    model_name.append('MEKT')
    base_clf4 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='MEKT-LDA', ete_method=None, pre_est=pre_est)
    clf.append(base_clf4)
    
    # 5 baseline RAVE
    model_name.append('RAVE')   
    base_clf5 = EL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='ABC-MDM', ete_method=None, njobs=njobs, 
        fs_new=fs)
    clf.append(base_clf5)
    
    # 6 baseline RAVE+
    model_name.append('RAVE+')
    base_clf6 = EL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='ABC-TS-LDA', ete_method=None, njobs=njobs, fs_new=fs)
    clf.append(base_clf6)
    
    # 7 baseline RA-EEGNet
    model_name.append('EEGNet')
    model = DL_Classifier(
        model_name='EEGNet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=300, 
        device='cuda')
    base_clf7 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method=None, ete_method=model, pre_est=pre_est)
    clf.append(base_clf7)
    
    # 8 baseline RA-LMDA-Net
    model_name.append('LMDA-Net')
    model = DL_Classifier(
        model_name='LMDANet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=300, 
        device='cuda')
    base_clf8 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method=None, ete_method=model, pre_est=pre_est)
    clf.append(base_clf8)
    
    # 9 baseline RA-Tensor-CSPNet
    model_name.append('Tensor-CSPNet')
    model = DL_Classifier(
        model_name='Tensor_CSPNet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=100, 
        device='cuda')
    base_clf9 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method=None, ete_method=model)
    clf.append(base_clf9)
    
    # ######################################### proposed #########################################
    
    # proposed RSFDA
    model_name.append('RSFDA')
    pro_clf00 = EL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='MEKT-P-MIC-K-LDA', ete_method=None, njobs=njobs, 
        fs_new=fs, cs_method='rsf', fea_num=30)
    clf.append(pro_clf00)
    
    ###################################### Ablation experiment ###################################
    
    # 1 sub-method RSFDA-1 
    model_name.append('RSFDA-1')
    sub_clf1 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='MEKT-P-MIC-K-LDA', ete_method=None, 
        pre_est=pre_est_rsf, fea_num=30)
    clf.append(sub_clf1)
    
    # 2 sub-method RSFDA-2 
    model_name.append('RSFDA-2')
    sub_clf2 = EL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='MEKT-P-MIC-K-LDA', ete_method=None, 
        njobs=njobs, fs_new=fs, fea_num=30)
    clf.append(sub_clf2)
    
    # 3 sub-method RSFDA-3 
    model_name.append('RSFDA-3')
    sub_clf3 = EL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method='TS', fes_method=None, 
        clf_method='LDA', end_method=None, ete_method=None, 
        njobs=njobs, fs_new=fs, cs_method='rsf')
    clf.append(sub_clf3)
    
    # 4 sub-method RSFDA-12 
    model_name.append('RSFDA-12')
    sub_clf4 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method=None, fes_method=None, 
        clf_method=None, end_method='MEKT-P-MIC-K-LDA', ete_method=None, 
        pre_est=pre_est, fea_num=30)
    clf.append(sub_clf4)
    
    # 5 sub-method RSFDA-13 
    model_name.append('RSFDA-13')
    sub_clf5 = TL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method='TS', fes_method=None, 
        clf_method='LDA', end_method=None, ete_method=None, pre_est=pre_est_rsf)
    clf.append(sub_clf5)
    
    # 6 sub-method RSFDA-23 
    model_name.append('RSFDA-23')
    sub_clf6 = EL_Classifier(
        target_domain=target_domain, dpa_method='RA', fee_method='TS', fes_method=None, 
        clf_method='LDA', end_method=None, ete_method=None, njobs=njobs, fs_new=fs)
    clf.append(sub_clf6)

    ########################################### END ##############################################
    
    for i, clf in enumerate(clf):
        # check if the model has been trained before
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                results = [json.loads(line) for line in f.readlines()]
                if any(result['model_name'] == model_name[i] for result in results):
                    print(f"Model {model_name[i]} has been trained, skip it.")
                    continue    
        
        # train the model and evaluate the performance
        print(f"Training model {model_name[i]}...")
        try:
            start_time = time.time()
            clf.fit(X_train, y_train)  
            train_time = time.time() - start_time
            start_time = time.time()
            score = clf.score(X_test, y_test)
            test_time = time.time() - start_time
        except Exception as e:
            print(f"Model {model_name[i]} failed: {e}")
            continue
        
        # save temporary results
        result = {'index': i, 'model_name': model_name[i],
                  'score': score, 'train_time': train_time, 'test_time': test_time, 
                  'target_domain': target_domain, 'fs': fs}
        with open(file_name, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Model {model_name[i]} score: {score}, train time: {train_time:.2f}s")
        

if __name__ == '__main__':
    
    for dataset_name in ['Pan2023', 'BNCI2014_001', 'BNCI2015_001']:
        
        # difine the ratio of target domain training samples
        ratio=0.2
        
        # define the result folder
        folder = f'./results_for_paper2024/cross_sessions_{ratio}_new/{dataset_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # initialize the dataset
        fs = 128 # sampling frequency
        datapath = 'datasets' # path to the dataset folder
        if dataset_name == 'BNCI2015_001':
            dataset = Dataset_MI(dataset_name,fs=fs,fmin=1,fmax=40,tmin=0,tmax=4,path=datapath)
        else:
            dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=1,fmax=40,tmin=0,tmax=4,path=datapath)
        subjects = dataset.subject_list
    
        # run the experiment for each subject
        for sub in subjects:
            print(f"Subject {sub}...")
            filename = folder + f'/result_sub{sub:02d}.json'
            
            # load data and split into source and target domains
            data, label, info = dataset.get_data([sub])
            session_values = info['session'].unique()
            print('the session values are:', session_values)
            session_indices = info.groupby('session').apply(lambda x: x.index.tolist())
            session_index_dict = dict(zip(session_values, session_indices))

            Data, Label=[], []
            for session in session_values[:2]:
                Data.append(data[session_index_dict[session]])
                Label.append(label[session_index_dict[session]])

            X, y_enc, domain =encode_datasets(Data, Label)
            print(f"data shape: {X.shape}, label shape: {y_enc.shape}")
            print(f"All Domain: {domain}")

            target_domain = domain[-1]
            idx_source = np.where(domain != target_domain)[0]
            idx_target = np.where(domain == target_domain)[0]
            
            print(f"Target domain: {target_domain}")
            tl_cv = TLSplitter(target_domain=target_domain, cv=ratio, no_calibration=False)

            # evaluate the model using the cross-validation splitter function
            for train, test in tl_cv.split(X, y_enc):
                X_train, y_train = X[train], y_enc[train]
                X_test, y_test = X[test], y_enc[test]
                models(X_train, y_train, X_test, y_test, 
                    target_domain=target_domain, fs=fs, 
                    file_name=filename, njobs=-1)
            

            
