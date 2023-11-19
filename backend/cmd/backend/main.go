package main

import (
	// "fmt"
	"log"
	// "time"
	"net"
	_"time"

	"golang.org/x/net/context"
	//import grpc
	"google.golang.org/grpc"
	
	
	"backend/internal/parse"
	"backend/pb"
)

type server struct{}


func (s *server) GetResumes(ctx context.Context, req *pb.ResumeRequest) (*pb.ResumeResponse, error) {
    // fmt.Println("Requested" + req.ID[0])
	resume_list := parse.Parse()

	//time.Sleep(10 * time.Second)

	response := &pb.ResumeResponse{}
	response.Resumes = resume_list

    return response, nil
}

func (s *server) SendResumes(ctx context.Context, res *pb.ResumeResponse) (*pb.ResumeRequest, error) {
	return &pb.ResumeRequest{}, nil
}


func main() {
	
	lis, err := net.Listen("tcp", ":5007")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// create grpc server
	opts := []grpc.ServerOption{}
	s := grpc.NewServer(opts...)
	
	// register server
	pb.RegisterResumeServiceServer(s, &server{})
	// start server
    s.Serve(lis)
}
