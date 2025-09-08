import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.feature_params = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            self.feature_params[c] = {
                "mean": X_c.mean(axis=0),
                "std": X_c.std(axis=0) + 1e-6
            }

    def _gaussian_likelihood(self, x, mean, std):
        return (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(- ((x - mean) ** 2) / (2 * std ** 2))

    def explain(x, vocab, mu0, var0, mu1, var1, pred, tfidf_matrix, labels):
        n_features = tfidf_matrix.shape[1]
        selected = []
        for i, tfidf in enumerate(x):
            if i >= n_features:
                continue
            if tfidf > 0 and vocab[i]:
                if pred == 1:
                    likelihood_pred = (1.0 / np.sqrt(2 * np.pi * var1[i])) * np.exp(-((tfidf - mu1[i]) ** 2) / (2 * var1[i]))
                    likelihood_other = (1.0 / np.sqrt(2 * np.pi * var0[i])) * np.exp(-((tfidf - mu0[i]) ** 2) / (2 * var0[i]))
                    diff = likelihood_pred - likelihood_other
                else:
                    likelihood_pred = (1.0 / np.sqrt(2 * np.pi * var0[i])) * np.exp(-((tfidf - mu0[i]) ** 2) / (2 * var0[i]))
                    likelihood_other = (1.0 / np.sqrt(2 * np.pi * var1[i])) * np.exp(-((tfidf - mu1[i]) ** 2) / (2 * var1[i]))
                    diff = likelihood_pred - likelihood_other
                selected.append((vocab[i], tfidf, likelihood_pred, diff))
        if pred == 1:
            selected.sort(key=lambda x: -x[3])
        else:
            selected.sort(key=lambda x: x[3])
        for token, tfidf, likelihood_pred, diff in selected:
            print(f"{token}\t{tfidf:.4f}\t{likelihood_pred:.6f}\t{diff:.6f}")

    def predict_proba(self, X):
        probs = []
        for x in X:
            nonzero_idx = np.where(x != 0)[0]
            class_probs = {}
            for c in self.classes:
                prior = self.class_priors[c]
                mean = self.feature_params[c]["mean"][nonzero_idx]
                std = self.feature_params[c]["std"][nonzero_idx]
                x_nonzero = x[nonzero_idx]
                likelihoods = self._gaussian_likelihood(x_nonzero, mean, std)
                log_likelihood = np.sum(np.log(likelihoods + 1e-9))
                class_probs[c] = np.log(prior) + log_likelihood
            max_log = max(class_probs.values())
            exp_probs = {c: np.exp(class_probs[c] - max_log) for c in self.classes}
            total = sum(exp_probs.values())
            probs.append({c: exp_probs[c] / total for c in self.classes})
        return probs

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.array([max(p, key=p.get) for p in probas])

if __name__ == "__main__":
    from scipy.sparse import load_npz
    import csv

    # Defina os caminhos dos arquivos diretamente no código
    tfidf_path = r"../SpamDetector/data/processed/tfidf_sparse.npz"
    csv_path = r"../SpamDetector/data/processed/emails_test.csv"
    vocab_path = r"../SpamDetector/data/processed/vocab.txt"

    # Carrega matriz TF-IDF de treino
    X = load_npz(tfidf_path).toarray()
    y = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y.append(int(row["spam"]))
    y = np.array(y)
    if X.shape[0] < y.shape[0]:
        y = y[:X.shape[0]]
    elif X.shape[0] > y.shape[0]:
        X = X[:y.shape[0]]

    # Carrega vocab
    with open(vocab_path, encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]

    print("Distribuição das classes:", np.bincount(y))
    print("Proporção de features não nulas:", np.count_nonzero(X) / X.size)

    # Treina e testa no próprio conjunto
    model = NaiveBayes()
    model.fit(X, y)
    preds = model.predict(X)

    print("Classe prevista:", preds)
    print("Classe real:", y)
    print("Acurácia:", np.mean(preds == y))