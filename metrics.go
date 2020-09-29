package naivebayes

import "errors"

// func unique(arr []int) (result []int) {
// 	occured := map[int]bool{}
// 	for e := range arr {
// 		if occured[arr[e]] != true {
// 			occured[arr[e]] = true
// 			result = append(result, arr[e])
// 		}
// 	}
// 	return
// }

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
