
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model, datasets
import pylab as pl


def draw_ROC(model, dtrain, dvalid, dtest, y_train, y_valid, y_test):
 probas_ = model.predict(dvalid, ntree_limit= model.best_ntree_limit)
 probas_1 = model.predict(dtrain, ntree_limit= model.best_ntree_limit)
 probas_2 = model.predict(dtest, ntree_limit= model.best_ntree_limit)

 fpr, tpr, thresholds = roc_curve(y_valid, probas_) # red
 fpr_1, tpr_1, thresholds_1 = roc_curve(y_train, probas_1)# blue
 fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, probas_2) # green

 roc_auc = auc(fpr, tpr)
 roc_auc_1 = auc(fpr_1, tpr_1)
 roc_auc_2 = auc(fpr_2, tpr_2)

 print("Area under the ROC curve - validation: %f" % roc_auc)
 print("Area under the ROC curve - train: %f" % roc_auc_1)
 print("Area under the ROC curve - test: %f" % roc_auc_2)

 # Plot ROC curve
 plt.figure(figsize=(8, 8))
 plt.plot(fpr, tpr, label='ROC curve - valid(AUC = %0.2f)' % roc_auc, color='r')
 plt.plot(fpr_1, tpr_1, label='ROC curve - train (AUC = %0.2f)' % roc_auc_1, color='b')
 plt.plot(fpr_2, tpr_2, label='ROC curve - test (AUC = %0.2f)' % roc_auc_2, color='g')
 plt.plot([0, 1], [0, 1], 'k--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.0])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('ROC for lead score model')
 plt.legend(loc="lower right")
 plt.show()
