


from ensemble_learning.el_classifier import EL_Classifier

# EL-RA-MDM-ABC (RAVE)
Rave = EL_Classifier(dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method='ABC-MDM', ete_method=None)

# EL-RA-TS-LR-ABC (RAVEplus)
Rave1 = EL_Classifier(dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method='ABC-TS-LR', ete_method=None)
