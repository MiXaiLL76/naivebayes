package naivebayes

import "errors"

//AccuracyScore Accuracy classification score.
func AccuracyScore(yTrue []int, yPred []int) (score float64, err error) {
	if len(yTrue) != len(yPred) {
		err = errors.New("len(yTrue) != len(yPred)")
		return
	}

	differingLabels := .0
	for i, val := range yTrue {
		if val-yPred[i] != 0 {
			differingLabels++
		}
	}
	score = 1 - differingLabels/float64(len(yTrue))
	return
}
