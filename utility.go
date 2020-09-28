package naivebayes

import "math"

// https://pyprog.pro/sort/argmax.html
func argmax(array []float64) (argmax int, maximum float64) {
	argmax = -1
	maximum = math.Inf(-1)
	for i, value := range array {
		if maximum < value {
			argmax, maximum = i, value
		}
	}
	return
}

// https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
func logsumexp(array []float64) float64 {
	_, aMax := argmax(array)
	if math.IsInf(aMax, 0) {
		aMax = 0
	}
	tmp := make([]float64, len(array))
	sum := .0
	for i, value := range array {
		tmp[i] = math.Exp(value - aMax)
		sum += tmp[i]
	}

	return math.Log(sum) + aMax
}
