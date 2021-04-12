package naivebayes

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
)

// GaussianNB class
type GaussianNB struct {
	Priors  []float64   `json:"priors"`
	Sigmas  [][]float64 `json:"sigmas"`
	Thetas  [][]float64 `json:"theta"`
	Classes []int       `json:"classes"`

	epsilon_      float64
	var_smoothing float64
	class_count   []float64
	class_prior_  []float64
}

//New build new GaussianNB class
// Returns
// -------
// GaussianNB{}
func New(priors []float64, sigmas [][]float64, thetas [][]float64, classes []int) (gnb *GaussianNB, err error) {
	if classes == nil {
		classes = make([]int, len(priors))
		for i := 0; i < len(classes); i++ {
			classes[i] = i
		}
	}

	if len(priors) != len(classes) {
		err = errors.New("len(priors) != len(classes)")
	}
	n, m := getShape(sigmas)
	a, b := getShape(thetas)
	if a != n || b != m {
		err = fmt.Errorf("sigmas.shape(%d, %d) != thetas.shape(%d, %d)", n, m, a, b)
	}
	log.Println()
	gnb = &GaussianNB{
		Priors:  priors,
		Sigmas:  sigmas,
		Thetas:  thetas,
		Classes: classes,
	}
	return
}

//GetWeight Get weight for this estimator.
// Returns
// -------
// A: JSON of GaussianNB class
// B: Error
func (gnb *GaussianNB) GetWeight() (b []byte, err error) {
	b, err = json.Marshal(gnb)
	return
}

//SetWeight set weight for this estimator.
// Parameters
// ----------
// I : JSON bytes for Unmarshal
// Returns
// -------
// A: Error
func (gnb *GaussianNB) SetWeight(I []byte) (err error) {
	err = json.Unmarshal(I, &gnb)
	return
}

func NewTrain(priors []float64, var_smoothing float64) (gnb *GaussianNB, err error) {
	gnb = &GaussianNB{
		Priors:        priors,
		var_smoothing: var_smoothing,
	}
	return
}

func (gnb *GaussianNB) partial_fit(X [][]float64, Y []float64, classes []int, refit bool, weight []float64) (err error) {
	localVariance := variance(X, 0)
	_, varMax := argmax(localVariance[0])
	gnb.epsilon_ = gnb.var_smoothing * varMax

	if refit {
		gnb.Classes = make([]int, 0)
	}
	if len(gnb.Classes) == 0 {
		gnb.Classes = classes
		_, nfeatures := getShape(X)
		nclasses := len(gnb.Classes)

		gnb.Thetas = make([][]float64, nclasses)
		for i := 0; i < nclasses; i++ {
			gnb.Thetas[i] = make([]float64, nfeatures)
		}

		gnb.Sigmas = make([][]float64, nclasses)
		for i := 0; i < nclasses; i++ {
			gnb.Sigmas[i] = make([]float64, nfeatures)
		}

		gnb.class_count = make([]float64, nclasses)

		if len(gnb.Priors) >= 0 {
			priors := gnb.Priors

			if len(priors) != nclasses {
				err = errors.New("Number of priors must match number of classes.")
				return
			}

			if arraySum(priors) != 1.0 {
				err = errors.New("The sum of the priors should be 1.")
				return
			}

			for _, p := range priors {
				if p < 0 {
					err = errors.New("Priors must be non-negative.")
					return
				}
			}
		} else {
			gnb.class_prior_ = make([]float64, len(gnb.Classes))
		}
	} else {
		_, s0 := getShape(X)
		_, s1 := getShape(gnb.Thetas)

		if s0 != s1 {
			err = errors.New(fmt.Sprintf("Number of features %d does not match previous data %d.", s0, s1))
		}

		for i, row := range gnb.Sigmas {
			for j := 0; j < len(row); j++ {
				gnb.Sigmas[i][j] -= gnb.epsilon_
			}
		}

	}

	classes = gnb.Classes
	yUnique := unique(Y)
	if !all(in1d(yUnique, int_as_float(classes))) {
		err = fmt.Errorf("The target label(s) %v in y do not exist in the initial classes %v", yUnique, classes)
		return
	}

	for _, yi := range yUnique {
		_, i := in_array(int(yi), classes)

		_, yi_in_y := all_in_array(yi, Y)
		X_i := make([][]float64, 0)

		sw_i := make([]float64, 0)
		for _, xi := range yi_in_y {
			X_i = append(X_i, X[xi])
			if len(weight) > 0 {
				sw_i = append(sw_i, weight[xi])
			}
		}
		var N_i float64
		if len(weight) > 0 {
			N_i = arraySum(sw_i)
		} else {
			N_if, _ := getShape(X_i)
			N_i = float64(N_if)
		}
		fmt.Println(N_i, i)
		// 431 : new_theta, new_sigma = self._update_mean_variance(
		// 	self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
		// 	X_i, sw_i)

	}
	return
}

func (gnb *GaussianNB) updateMeanVariance(n_past int, mu []float64, _var []float64, X [][]float64, weight []float64) (total_mu []float64, total_var []float64) {
	s, _ := getShape(X)
	if s == 0 {
		return mu, _var
	}

	if len(weight) > 0 {
		n_new := arraySum(weight)
		fmt.Println(n_new)
	}
	return
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

//Score Return the mean accuracy on the given test data and labels.
func (gnb *GaussianNB) Score(X [][]float64, y []int) (score float64, err error) {
	pred := gnb.Predict(X)
	score, err = AccuracyScore(y, pred)
	return
}
