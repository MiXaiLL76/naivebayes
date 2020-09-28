package naivebayes

import "math"

// GaussianNB class
type GaussianNB struct {
	Priors  []float64   `json:"priors"`
	Sigmas  [][]float64 `json:"sigmas"`
	Thetas  [][]float64 `json:"theta"`
	Classes []int       `json:"classes"`
}

//New Создание нового класса GaussianNB с параметрами
func New(priors []float64, sigmas [][]float64, thetas [][]float64, classes []int) *GaussianNB {
	if classes == nil {
		classes = make([]int, len(priors))
		for i := 0; i < len(classes); i++ {
			classes[i] = i
		}
	}

	return &GaussianNB{
		Priors:  priors,
		Sigmas:  sigmas,
		Thetas:  thetas,
		Classes: classes,
	}
}

func (gnb *GaussianNB) jointLoglikelihood(features []float64) []float64 {
	likelihoods := make([]float64, len(gnb.Sigmas))

	for i, sigmas := range gnb.Sigmas {
		sum := 0.
		for _, sigma := range sigmas {
			sum += math.Log(2. * math.Pi * sigma)
		}
		nij := -0.5 * sum

		sum = 0.
		for j, sigma := range sigmas {
			sum += math.Pow(features[j]-gnb.Thetas[i][j], 2.) / sigma
		}
		nij -= 0.5 * sum

		likelihoods[i] = math.Log(gnb.Priors[i]) + nij
	}

	return likelihoods
}

//PredictLogProba return log-probability estimates for the test vector X.
// Parameters
// ----------
// X : array-like of shape (n_samples, n_features)
// Returns
// -------
// C : array-like of shape (n_samples, n_classes)
// 	Returns the log-probability of the samples for each class in
// 	the model. The columns correspond to the classes in sorted
// 	order, as they appear in the attribute :term:`classes_`.
func (gnb *GaussianNB) PredictLogProba(X [][]float64) (predict [][]float64) {
	for _, item := range X {
		likelihoods := gnb.jointLoglikelihood(item)
		LOG := logsumexp(likelihoods)

		for i := 0; i < len(likelihoods); i++ {
			likelihoods[i] -= LOG
		}

		predict = append(predict, likelihoods)
	}

	return
}

//PredictProba Return probability estimates for the test vector X.
// Parameters
// ----------
// X : array-like of shape (n_samples, n_features)
// Returns
// -------
// C : array-like of shape (n_samples, n_classes)
// 	Returns the probability of the samples for each class in
// 	the model. The columns correspond to the classes in sorted
// 	order, as they appear in the attribute :term:`classes_`.
func (gnb *GaussianNB) PredictProba(X [][]float64) (predict [][]float64) {
	logPredict := gnb.PredictLogProba(X)

	for _, feature := range logPredict {
		proba := make([]float64, len(feature))
		for i := 0; i < len(feature); i++ {
			proba[i] = math.Exp(feature[i])
		}
		predict = append(predict, proba)
	}
	return
}

//Predict Perform classification on an array of test vectors X.
// Parameters
// ----------
// X : array-like of shape (n_samples, n_features)
// Returns
// -------
// C : ndarray of shape (n_samples,)
// 	Predicted target values for X
func (gnb *GaussianNB) Predict(X [][]float64) (predict []int) {
	proba := gnb.PredictProba(X)
	for _, probability := range proba {
		i, _ := argmax(probability)
		predict = append(predict, gnb.Classes[i])
	}
	return
}
