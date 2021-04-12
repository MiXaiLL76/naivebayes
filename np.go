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

func unique(arr []float64) (result []float64) {
	occured := map[float64]bool{}
	for e := range arr {
		if occured[arr[e]] != true {
			occured[arr[e]] = true
			result = append(result, arr[e])
		}
	}
	return
}

func in1d(arr []float64, classes []float64) (result []bool) {
	for _, v := range arr {
		ok, _ := in_array(v, classes)
		result = append(result, ok)
	}
	return
}

func all(arr []bool) bool {
	for _, v := range arr {
		if v == false {
			return false
		}
	}
	return true
}

func getShape(array [][]float64) (samples int, classes int) {
	samples = len(array)
	if samples > 0 {
		classes = len(array[0])
		for _, sub := range array {
			if classes != len(sub) {
				classes = -1
				break
			}
		}

	}
	return
}

// variance methods
func arraySum(array []float64) (result float64) {
	for _, v := range array {
		result += v
	}
	return
}

func trueDivide(array [][]float64, div float64) (out [][]float64) {
	out = make([][]float64, len(array))
	for i, row := range array {
		out[i] = make([]float64, len(row))
		for j, col := range row {
			out[i][j] = col / div
		}
	}
	return
}

func umrSum(array [][]float64, axis interface{}) (sum [][]float64) {
	sum = make([][]float64, 1)
	switch axis {
	case 0:
		sum[0] = make([]float64, len(array[0]))
		for _, row := range array {
			for i, col := range row {
				sum[0][i] += col
			}
		}
	case 1:
		for _, row := range array {
			sum[0] = append(sum[0], arraySum(row))
		}
	default:
		sum[0] = make([]float64, 1)
		for _, row := range array {
			for _, col := range row {
				sum[0][0] += col
			}
		}
	}

	return
}

// https://numpy.org/doc/stable/reference/generated/numpy.var.html
func variance(array [][]float64, axis interface{}) (ret [][]float64) {
	samples, classes := getShape(array)

	var rcount int
	switch axis {
	case 0:
		rcount = samples
	case 1:
		rcount = classes
	default:
		rcount = samples * classes
	}

	arrmean := umrSum(array, axis)
	arrmean = trueDivide(arrmean, float64(rcount))
	// fmt.Println("arrmean", arrmean)

	x := make([][]float64, len(array))
	for i, row := range array {
		x[i] = make([]float64, len(row))
		for j, col := range row {
			// fmt.Println("X", x)
			switch axis {
			case 0:
				x[i][j] = math.Pow((col - arrmean[0][j]), 2)
			case 1:
				x[i][j] = math.Pow((col - arrmean[0][i]), 2)
			default:
				x[i][j] = math.Pow((col - arrmean[0][0]), 2)
			}

		}
	}
	// fmt.Println("X", x)

	ret = umrSum(x, axis)
	_, div2 := argmax([]float64{float64(rcount), 0})
	ret = trueDivide(ret, div2)

	return
}
