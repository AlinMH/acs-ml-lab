import logging

import cvxopt.solvers
import numpy as np

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVMTrainer(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def fit(self, X, y):
        """
        Dandu-se setul de antrenare X si label-urile y, intoarce un SVM antrenat.
        :param X: setul de antrenare avand dimensiunea (num_training_points, num_features)
        :param y: eticheta fiecarui input din setup de antrenare, avand dimensiunea (num_training_points, 1)
        :return: Predictorul SVM antrenat.
        """
        ## Pas 1 - calculeaza multiplicatorii Lagrange, rezolvand problema duala
        lagrange_multipliers = self._compute_multipliers(X, y)

        ## Pas 2 - intoarce predictorul SVM pe baza multiplicatorilor Lagrange
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):
        """
        Precalculeaza matricea Gram, folosind kernel-ul dat in constructor, in vederea rezolvarii problemei duale.
        :param X: setul de date de antrenare avand dimesiunea (num_samples, num_features)
        :return: Matricea Gram precalculata
        """
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])

        # TODO: populati matricea Gram conform kernel-ului selectat

        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        ## Desi bias-ul poate fi calculat pe baza unei singure valori din setul vectorilor de suport,
        # pentru o forma stabila numeric, folosim o media peste toti vectorii suport
        bias = np.mean(
            [
                y_k
                - SVMPredictor(
                    kernel=self._kernel,
                    bias=0.0,
                    weights=support_multipliers,
                    support_vectors=support_vectors,
                    support_vector_labels=support_vector_labels,
                ).predict(x_k)
                for (y_k, x_k) in zip(support_vector_labels, support_vectors)
            ]
        )

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels,
        )

    def _compute_multipliers(self, X, y):
        """
        Rezolva problema de optimizare duala, calculand valoarea multiplicatorilor Lagrange
        :param X: setul de date de antrenare avand dimensiunea (num_samples, num_features)
        :param y: setul de etichete pentru datele de antrenare avand dimensiunea (num_samples, 1)
        :return: lista multiplicatorilor Lagrange
        """
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        """
        Metoda din cvxopt ce rezolva o problema de optimizare patratica are urmatoarea formulare
        min 1/2 x^T P x + q^T x
        a.i.
        Gx < h
        Ax = b
        
        unde x este vectorul de valori x_i, de dimensiune (n_samples, 1) a carui valoare se cauta
        """

        """
        Problema duala pentru SVM cere:
        min 1/2 a^T Q a - 1^T a
        a.i.
        0 <= a_i, orice i
        a_i <= C, orice i
        y^T a = 0

        unde Q = (y * y^T) . K (i.e. inmultire matriceala intre y si y^T si apoi inmultire element-cu-element cu matricea K)

        Aici vectorul pe care il cautam este `a' (cel al multiplicatorilor Lagrange) de dimensiune (n_samples, 1).
        """

        """
        Cerinta este de a gasi maparea corecta intre forma duala pentru SVM si cea utilizata de cvxopt, i.e. ce valori trebuie sa ia
        matricile P, G si A si vectorii q, h si b, astfel incat ei sa reprezinte expresiile din forma duala SVM.
        
        Vectorul `a' tine loc de `x' in forma ecuatiilor pentru cvxopt.
        """

        # TODO calculeaza valoarea matricii P = (y * y^T) . K
        P = cvxopt.matrix(np.multiply(y * np.transpose(y), K))

        # TODO calculeaza valoarea vectorului q
        q = cvxopt.matrix(np.ones(n_samples))

        # setam G si h in doi pasi a.i sa cuprinda cele doua inegalitati, 0 <= a_i si a_i <= C
        # TODO seteaza G_std si h_std pentru a rezolva -a <= 0
        G_std = cvxopt.matrix(-1 * np.identity(n_samples))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # TODO seteaza G_slack si h_slack pentru a rezolva a <= C
        G_slack = cvxopt.matrix(np.identity(n_samples))
        h_slack = cvxopt.matrix(self._c * np.ones(n_samples))

        # TODO obtine G si h prin suprapunere a variabilelor anterioare (vezi functia numpy.vstack)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        # TODO seteaza A si b a.i. sa acopere relatia y^T a = \sum y_i a_i
        A = cvxopt.matrix(np.diag(y), tc="d")
        b = cvxopt.matrix(np.zeros(n_samples))

        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)  # decomentati linia cand ati implementat matricile de mai sus

        # intoarcem multiplicatorii Lagrange sub forma liniarizata - vezi functia np.ravel
        return np.ravel(solution["x"])  # decomentati linia cand ati implementat matricile de mai sus


class SVMPredictor(object):
    def __init__(self, kernel, bias, weights, support_vectors, support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)
        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)

    def predict(self, x):
        """
        Calculeaza predictia facuta de un SVM, dandu-se inputul x.
        Formula de calcul este:
            \sum_m [z_m * y_m * kernel(x_m, x)] + bias

            unde m itereaza peste multimea vectorilor de suport
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights, self._support_vectors, self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()

    def score(self, X_test, y_test):
        nr_samples, nr_dim = X_test.shape

        predictions = np.array(map(self.predict, X_test))
        matches = np.multiply(predictions, y_test)

        score = sum(matches[matches == 1], 0) * 1.0 / nr_samples
        return score
