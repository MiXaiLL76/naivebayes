package naivebayes

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// https://github.com/stretchr/testify

func TestVariance(t *testing.T) {
	input := [][]float64{{1, 2, 3, 4, -10, 1}, {1, 2, 3, 4, -10, 2}}

	output_nil := [][]float64{{3}}
	assert.Equal(t, output_nil, umrSum(input, nil))

	output0 := [][]float64{{2, 4, 6, 8, -20, 3}}
	assert.Equal(t, output0, umrSum(input, 0))

	output1 := [][]float64{{1, 2}}
	assert.Equal(t, output1, umrSum(input, 1))

	output2 := [][]float64{{21.805555555555557, 22.222222222222225}}
	assert.Equal(t, output2, variance(input, 1))

	output3 := [][]float64{{0, 0, 0, 0, 0, 0.25}}
	assert.Equal(t, output3, variance(input, 0))

	output4 := [][]float64{{22.020833333333332}}
	assert.Equal(t, output4, variance(input, nil))
}

func TestArgmax(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1}
	output := 5

	a, b := argmax(input)

	assert.Equal(t, a, output)
	assert.Equal(t, b, input[output])
}
func TestAccuracyScore(t *testing.T) {
	score, err := AccuracyScore([]int{0, 1, 2, 3, 4}, []int{0, 1, 2, 3, 5})
	assert.Nil(t, err)
	assert.Equal(t, score, 0.8)

	_, err = AccuracyScore([]int{0, 1, 2, 3}, []int{0, 1, 2, 3, 5})
	assert.Error(t, err)
}
func TestLogSumExp(t *testing.T) {
	input := []float64{-1525, -981}
	output := -981.0
	assert.Equal(t, logsumexp(input), output)
}

func TestGaussianNB(t *testing.T) {
	priors := []float64{0.64, 0.36}
	sigmas := [][]float64{
		{
			0.531313959436933, 0.3676670982424953, 0.3261662484173987,
			0.14630583259786364, 0.03454532922427119, 0.07618212725869644,
			0.07095442906286838, 3.4023372937440985, 0.06953857729852864,
			3.1679531933463623, 0.2271729150215087, 0.010047478230326092,
			0.06408217033252246, 3.8345589109308773, 0.08245527911041908,
			3.5857255075492267, 0.23716024686797907, 0.008947301026864501,
		},
		{
			0.016751786089326537, 0.22298274743124824, 0.06362243330891189,
			0.06272662991927497, 0.022632718860116585, 0.0512284815079035,
			0.4789906049871742, 0.7856600346876715, 0.02666296203750278,
			0.7568680367211857, 0.4266804161261409, 0.06584440794624873,
			0.16617672733500646, 1.0075916247551457, 0.36634894364542536,
			0.8697764986842066, 0.2321303137469273, 0.07473562977893462,
		},
	}
	thetas := [][]float64{
		{
			-0.378030418125, 0.92968765125, -0.0214451675,
			1.49651614, 0.35658525062499996, 0.5668284881250001,
			-0.6317826125, -3.8688783824999993, 0.233494285,
			-3.52284107875, 0.8652768975, 0.34603730625,
			-0.6409938531249999, -4.041267548749999, 0.24099661375,
			-3.703544053125, 0.88199046625, 0.337723499375,
		},
		{
			0.2328433577777778, -3.808441925555555, 0.8516475322222221,
			-2.1799084122222223, 0.6188041755555554, 1.6285335166666668,
			-2.9540502077777777, -6.12924791, -0.5442245622222223,
			-5.322713128888889, 2.4098256466666665, 0.8065347811111111,
			-2.3469731533333333, -6.500740411111111, -1.1397898011111112,
			-5.76542109, 1.2071833533333332, 0.7353193222222223,
		},
	}

	gnb, err := New(priors, sigmas, thetas, nil)
	assert.Nil(t, err)

	features := [][]float64{{-4.42682927, -2.75730994, -2.4097561, -1.89181287, 2.01707317, 0.86549708, -4.63902439, -6.83918129, 0.43414634, -3.29532164, 5.07317073, 3.54385965, -4.63902439, -6.83918129, 0.43414634, -3.29532164, 5.07317073, 3.54385965}}

	assert.Equal(t, gnb.jointLoglikelihood(features[0]), []float64{-1525.2351690216494, -981.4595836362245})

	assert.Equal(t, gnb.PredictLogProba(features), [][]float64{{-543.775585385425, 0}})

	assert.Equal(t, gnb.PredictProba(features), [][]float64{{6.938472532856835e-237, 1}})

	assert.Equal(t, gnb.Predict(features), []int{1})
}
