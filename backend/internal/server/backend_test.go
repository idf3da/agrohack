package server

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"testing"
)

func TestJSONPrint(t *testing.T) {
	js, err := os.Open("miserables.json")
	if err != nil{
		log.Fatal(err)
	}
	var date LinksTable
	if err := json.NewDecoder(js).Decode(&date); err!=nil{
		log.Fatal(err)
	}
	/*for _,value := range date.Nodes{
		fmt.Printf("Node{Id: %s, Group: %d}\n", value.Id, value.Group)
	}
	for _,value := range date.Links{
		fmt.Printf("Link{Source: %s, Target: %s, Value: %d}\n", value.Source, value.Target, value.Value)
	}*/

	data, _ := json.Marshal(date)
	fmt.Printf("%s\n", data)

}