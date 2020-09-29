package main

import (
	"encoding/json"
	"io/ioutil"
	"log"

	"github.com/MiXaiLL76/naivebayes"
)

// Data class
type Data struct {
	Features   [][]float64 `json:"features"`
	Prediction []int       `json:"prediction"`
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	gnb := naivebayes.GaussianNB{}
	weight, err := ioutil.ReadFile("predictor.json")
	checkErr(err)

	err = gnb.SetWeight(weight)
	checkErr(err)

	bytesData, err := ioutil.ReadFile("data.json")
	checkErr(err)

	var testData Data
	err = json.Unmarshal(bytesData, &testData)
	checkErr(err)

	prediction := gnb.Predict(testData.Features)

	pass := 0
	for i, val := range testData.Prediction {
		if val == prediction[i] {
			pass++
		}
	}
	if len(prediction) == pass {
		log.Println("Test passed")
	} else {
		log.Println("Test not passed")
	}
}
