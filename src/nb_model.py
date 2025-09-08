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

    def explain(self, x, vocab=None):
        """
        Explica a decisão do modelo para um único vetor x.
        Retorna um dicionário detalhado com todos os passos do cálculo.
        """
        explanation = {}
        nonzero_idx = np.where(x != 0)[0]
        words = [vocab[i] for i in nonzero_idx] if vocab is not None else nonzero_idx.tolist()
        x_nonzero = x[nonzero_idx]
        explanation['words'] = words
        explanation['tfidf'] = x_nonzero.tolist()
        explanation['classes'] = {}
        for c in self.classes:
            prior = self.class_priors[c]
            mean = self.feature_params[c]["mean"][nonzero_idx]
            std = self.feature_params[c]["std"][nonzero_idx]
            likelihoods = self._gaussian_likelihood(x_nonzero, mean, std)
            log_likelihoods = np.log(likelihoods + 1e-9)
            class_logprob = np.log(prior) + np.sum(log_likelihoods)
            explanation['classes'][int(c)] = {
                "prior": float(prior),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "likelihoods": likelihoods.tolist(),
                "log_likelihoods": log_likelihoods.tolist(),
                "log_likelihood_sum": float(np.sum(log_likelihoods)),
                "class_logprob": float(class_logprob)
            }
        return explanation

    def predict_proba(self, X):
        probs = []
        for x in X:
            nonzero_idx = np.where(x != 0)[0]  # Só considera features presentes no email
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

    # Caminhos dos arquivos
    tfidf_path = r"../SpamDetector/data/processed/tfidf_sparse.npz"
    csv_path = r"../SpamDetector/data/processed/emails_cleaned.csv"

    # Carrega matriz TF-IDF
    tfidf = load_npz(tfidf_path).toarray()

    # Carrega rótulos
    labels = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["spam"]))
    labels = np.array(labels)

    if tfidf.shape[0] < labels.shape[0]:
        labels = labels[:tfidf.shape[0]]
    elif tfidf.shape[0] > labels.shape[0]:
        tfidf = tfidf[:labels.shape[0]]

    # Separa último exemplo para teste
    n_test = 100
    X_train = tfidf[:-n_test]
    y_train = labels[:-n_test]
    X_test = tfidf[-n_test:]
    y_test = labels[-n_test:]

    print("Distribuição das classes no treino:", np.bincount(y_train))
    print("Distribuição das classes no teste:", np.bincount(y_test))
    print("Proporção de features não nulas no teste:", np.count_nonzero(X_test) / X_test.size)

    # Treina e testa
    model = NaiveBayes()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Mostra resultados
    print("Classe prevista:", preds)
    print("Classe real:", y_test)
    print("Acurácia:", np.mean(preds == y_test))