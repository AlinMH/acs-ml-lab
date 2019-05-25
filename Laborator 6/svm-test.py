import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

import svm
import datetime
from kernel import linear, radial_basis

if __name__ == "__main__":
    # h = .02  # step size in the mesh
    h = .1  # step size in the mesh

    # X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
    #                            random_state=1, n_clusters_per_class=1)
    # rng = np.random.RandomState(2)
    # X += 3 * rng.uniform(size=X.shape)
    # linearly_separable = (X, y)

    num_samples = 60
    num_features = 2
    X = np.matrix(np.random.normal(size=num_samples * num_features)
                        .reshape(num_samples, num_features))
    y = np.ravel(2 * (X.sum(axis=1) > 0) - 1.0)
    linearly_separable = (X, y)

    names = ["SVM Linear c = 1", "SVM RBF c = 1", "SVM Linear c = 100", "SVM RBF c = 100"]
    datasets = [make_moons(noise=0.1, random_state=0),
                make_circles(noise=0.1, factor=0.5, random_state=1),
                linearly_separable
                ]

    classifiers = [
        svm.SVMTrainer(kernel=linear(), c = 1),
        svm.SVMTrainer(kernel=radial_basis(gamma=1), c = 1),
        svm.SVMTrainer(kernel=linear(), c=100),
        svm.SVMTrainer(kernel=radial_basis(gamma=1), c=100)
    ]

    figure = plt.figure(figsize=(27, 9))
    i = 1
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        ## modify labels to correspond to SVM categories {-1, 1}
        y[y == 0] = -1

        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.34, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            
            predictor = clf.fit(X_train, y_train)
            score = predictor.score(X_test, y_test)

            mesh = np.c_[xx.ravel(), yy.ravel()]
            #print mesh

            start_ts = datetime.datetime.now()
            Z = np.array(map(predictor.predict, mesh))
            Z = Z.reshape(xx.shape)
            end_ts = datetime.datetime.now()
            diff = end_ts - start_ts
            print ("SVM mesh for dataset %s and classifier %s --> %s s." % (ds_cnt, name, diff.seconds))


            ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker = '+', cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker = '+', cmap=cm_bright, alpha=0.6)

            # Plot the support vectors
            ax.scatter(predictor._support_vectors[:, 0], predictor._support_vectors[:, 1], marker = 'v', c=predictor._support_vector_labels, cmap=cm_bright)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()
