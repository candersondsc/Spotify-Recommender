from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

def cross_validation(model, X, y, cv):
    """
    Cross-Validation ---> (0.5 Points)
    Perform k-fold cross-validation with support for multi-genre data
    """
    
    #Extract primary genres for model training
    primary_y = y.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    
    #Initialize KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    #Store scores
    cv_scores = []
    multi_genre_cv_scores = []
    
    #Perform manual cross-validation to compute both metrics
    for train_idx, test_idx in kf.split(X):
        #Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        primary_train = primary_y.iloc[train_idx]
        
        #Train model on primary genres
        model.fit(X_train, primary_train)
        
        #Predictions
        y_pred = model.predict(X_test)
        
        #Calculate traditional accuracy
        primary_test = primary_y.iloc[test_idx]
        fold_score = (y_pred == primary_test).mean()
        cv_scores.append(fold_score)
        
        #Calculate multi-genre accuracy
        correct = 0
        for true_genres, pred_genre in zip(y_test.values, y_pred):
            if isinstance(true_genres, list):
                if pred_genre in true_genres:
                    correct += 1
            else:
                if pred_genre == true_genres:
                    correct += 1
        
        multi_genre_score = correct / len(y_test)
        multi_genre_cv_scores.append(multi_genre_score)
    
    #Print results
    print(f"Cross-validation scores (primary genre): {cv_scores}")
    print(f"Mean CV score (primary genre): {np.mean(cv_scores):.4f}")
    print(f"Standard deviation: {np.std(cv_scores):.4f}")
    
    print(f"Multi-genre CV scores: {multi_genre_cv_scores}")
    print(f"Mean multi-genre CV score: {np.mean(multi_genre_cv_scores):.4f}")
    print(f"Standard deviation: {np.std(multi_genre_cv_scores):.4f}")
    
    return cv_scores, multi_genre_cv_scores

def plot_roc_curve(model, X, y, filename):
    #ROC/AUC --> 1 Point

    #Binarize the output for multi-class
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    
    #Use OneVsRestClassifier for multi-class
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X, y_bin).predict_proba(X)
    
    #Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    #Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title('Multi-class Genre ROC Curve', fontsize=30, pad=20)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show() 

    return fpr, tpr, roc_auc
