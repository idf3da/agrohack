package main 

import (
	"encoding/json"
	"log"
	"net/http"
	"github.com/gorilla/mux"
	_ "github.com/gorilla/handlers"
	"github.com/rs/cors"
	"os"
)

type Node struct {
	Id string 		`json:"id"`
	Group int 		`json:"group"`
}

type Link struct {
	Source string 	`json:"source"`
	Target string 	`json:"target"`
	Value int 		`json:"value"`
}

type linksTable struct {
	Nodes []Node	`json:"nodes"`
	Links []Link 	`json:"links"`
}

var LinksTable linksTable

func getLinksTable(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(LinksTable)
}

func main(){
	js, err := os.Open("miserables.json")
	if err != nil{
		log.Fatal(err)
	}
	if err := json.NewDecoder(js).Decode(&LinksTable); err!=nil{
		log.Fatal(err)
	}
	r := mux.NewRouter()
	r.HandleFunc("/linksTables", getLinksTable).Methods("GET")
	
	/*headersOk := handlers.AllowedHeaders([]string{"X-Requested-With"})
	originsOk := handlers.AllowedOrigins([]string{"*"})
	methodsOk := handlers.AllowedMethods([]string{"GET", "HEAD", "POST", "PUT", "OPTIONS"})
	*/
	c := cors.New(cors.Options{
        AllowedOrigins: []string{"*"},
		AllowedMethods:   []string{http.MethodGet, http.MethodPost, http.MethodDelete},		
        AllowCredentials: true,
 })

    handler := c.Handler(r)
	
	log.Fatal(http.ListenAndServe(":8000", handler))
}