# using grpc on :54321 listen for proto messages and print them
import grpc

import os
import sys

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_directory)

import resume_pb2 as pb
import resume_pb2_grpc as pb_grpc

class Resume(pb_grpc.ResumeServiceServicer):
    def GetResumes(self, request, context):
        print(request)
        return pb.ResumeResponse()

def get_resumes():
    res_data = []
    options = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel('127.0.0.1:5007', options = options)
    stub = pb_grpc.ResumeServiceStub(channel)
    # Create a ResumeRequest message
    request = pb.ResumeRequest(
        ID = "1"
    )

    # Send the ResumeRequest to the server and receive the response with context
    response = stub.GetResumes(request=request, )

    # Print the received response
    print("Received Resumes from Server:")
    for resume in response.resumes:
        for i in resume.Experience:
            res_.append(i)
    print(res_data)        

if __name__ == '__main__':
    res_ = get_resumes()