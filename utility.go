package naivebayes

import (
	"math"
	"reflect"
)

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

func in_array(val interface{}, array interface{}) (exists bool, index int) {
	exists = false
	index = -1

	switch reflect.TypeOf(array).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(array)

		for i := 0; i < s.Len(); i++ {
			if reflect.DeepEqual(val, s.Index(i).Interface()) == true {
				index = i
				exists = true
				return
			}
		}
	}

	return
}

func all_in_array(val interface{}, array interface{}) (exists bool, indexes []int) {
	exists = false

	switch reflect.TypeOf(array).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(array)

		for i := 0; i < s.Len(); i++ {
			if reflect.DeepEqual(val, s.Index(i).Interface()) == true {
				indexes = append(indexes, i)
				exists = true
			}
		}
	}

	return
}

func int_as_float(val []int) (out []float64) {
	for _, v := range val {
		out = append(out, float64(v))
	}
	return
}
